export CUDA_VISIBLE_DEVICES=1

model_name="SegRNN"
seed=${1:-123}



# 根据每个预测长度合理设置 seg_len，避免 reshape 报错
declare -A seg_lens
seg_lens["Yearly"]=6
seg_lens["Quarterly"]=4
seg_lens["Monthly"]=6
seg_lens["Weekly"]=13
seg_lens["Daily"]=7
seg_lens["Hourly"]=24



for optimizer in "TS_Adam" "Adam" "SGD" "Yogi" "AdamW" "lookahead_Adam"; do
  for seasonal_patterns in  'Hourly' ; do
    python -u run.py \
      --optimizer "$optimizer" \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns $seasonal_patterns \
      --model_id "m4_$seasonal_patterns" \
      --model $model_name \
      --data m4 \
      --features M \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --seg_len ${seg_lens[$seasonal_patterns]} \
      --d_model 128 \
      --dropout 0.2 \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --fix_seed "$seed" \
      --batch_size 8 \
      --train_epochs 20
  done
done