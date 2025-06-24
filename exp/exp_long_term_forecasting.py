from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import csv
import pandas as pd


from Optimizer.TS_Adam import TS_Adam
from Optimizer.Yogi import Yogi
from Optimizer.lookahead import Lookahead
from Optimizer.TS_Yogi import TS_Yogi
from Optimizer.TS_AdamW import TS_AdamW
from Optimizer.AutoCyclic import AutoCyclicLR


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self,  load_best = False):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if load_best:
            model.load_state_dict(torch.load(os.path.join('./best_model', self.args.model, self.args.data, str(self.args.pred_len),'checkpoint.pth')))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        scheduler = None
        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), betas=(0.9,0.999), lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'TS_Adam':
            model_optim = TS_Adam(self.model.parameters(), betas=(0.9,0.999), lr=self.args.learning_rate)
        elif self.args.optimizer == 'Yogi':
            model_optim = Yogi(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'lookahead_Adam':
            in_opt = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = Lookahead(in_opt, k=5, alpha=0.5)
        elif self.args.optimizer == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)

        elif self.args.optimizer == 'lookahead_TS_Adam':
            in_opt = TS_Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = Lookahead(in_opt, k=5, alpha=0.5)
        elif self.args.optimizer == 'TS_Yogi':
            model_optim = TS_Yogi(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'TS_AdamW':
            model_optim = TS_AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)

        elif self.args.optimizer == 'AutoCyclic':
            model_optim = optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=self.args.learning_rate)
            base_lr = 0.0002782559402207126
            max_lr_multiplier = 7
            cycle_size = 25
            scheduler = AutoCyclicLR(model_optim, base_lr=base_lr, max_lr=(base_lr * max_lr_multiplier),
                                     step_size=cycle_size)

        return model_optim, scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion, load_best = False):
        total_loss = []
        model = self.best_model if load_best else self.model
        if not load_best:
            model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        if not load_best:
            model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            train_loss_csv, val_loss_csv, test_loss_csv = [], [], []
            best_loss_csv = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if self.args.optimizer == 'AutoCyclic':
                        scheduler.step()

                # Output loss to CSV file
                if self.args.save_each_loss and (((i + 1) % 5 == 0) or i <= 5):
                    train_loss_csv.append(loss.item())
                    val_loss_csv.append(self.vali(vali_data, vali_loader, criterion).item())
                    test_loss_csv.append(self.vali(test_data, test_loader, criterion).item())

                # calculate regret
                if self.args.calculate_regret and ((i+1) % 5 == 0):
                    best_loss_csv.append(self.vali(test_data, test_loader, criterion, load_best = True).item())
                    if not self.args.save_each_loss:
                        test_loss_csv.append(self.vali(test_data, test_loader, criterion).item())



            if self.args.save_each_loss:

                # write csv file
                csv_file = self.args.loss_csv_fire

                # 检查文件是否已存在（决定是否写表头）
                write_header = not os.path.exists(csv_file)

                df = pd.DataFrame({
                    "number": range(1,len(train_loss_csv)+1),  # 自动生成连续的epoch编号: 0, 1, 2,...
                    "train_loss": train_loss_csv,
                    "val_loss": val_loss_csv,
                    "test_loss": test_loss_csv
                })
                # 批量写入CSV
                df.to_csv(
                    csv_file,
                    mode='a',  # 追加模式
                    header=write_header,  # 如果是新文件，write_header=True会写入表头
                    index=False,  # 不保存DataFrame的索引
                    encoding='utf-8'
                )


            if self.args.calculate_regret:
                # write csv file
                csv_file = self.args.regret_csv_fire

                # 检查文件是否已存在（决定是否写表头）
                write_header = not os.path.exists(csv_file)

                df = pd.DataFrame({
                    "number": range(1, len(best_loss_csv) + 1),  # 自动生成连续的epoch编号: 0, 1, 2,...
                    "best_loss": best_loss_csv,
                    "current_loss": test_loss_csv,
                })
                # 批量写入CSV
                df.to_csv(
                    csv_file,
                    mode='a',  # 追加模式
                    header=write_header,  # 如果是新文件，write_header=True会写入表头
                    index=False,  # 不保存DataFrame的索引
                    encoding='utf-8'
                )





            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            #####################################
            # 每轮都需要保存一个备份，来选择路径上的最佳模型
            # torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            torch.save(self.model.state_dict(), os.path.join(path, 'test_loss->{}'.format(test_loss) + 'checkpoint.pth'))

            #####################################
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
