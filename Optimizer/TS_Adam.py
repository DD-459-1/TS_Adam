



import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class TS_Adam(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # 如果 lr 没有传递，则抛出异常
        if lr is required:
            raise ValueError("Learning rate is required")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(TS_Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        # 在 step 时执行参数更新的操作
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化参数的状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                # 增加 weight decay 正则化
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state['step'] += 1
                t = state['step']


                # 计算一阶矩和二阶矩
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                # 偏差修正
                # 取消二阶偏差纠正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1


                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # 参数更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


        return loss
