export CUDA_VISIBLE_DEVICES=0

model_name="SegRNN"
seed=${1:-123}


for pred_len in 96 192 336 720  ; do
  for optimizer in "AutoCyclic" ; do
    python -u run.py \
      --optimizer "$optimizer" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "./dataset/ETT-small/" \
      --data_path "ETTm2.csv" \
      --model_id "ETTm2_96_$pred_len" \
      --model "$model_name" \
      --data ETTm2 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --seg_len 24 \
      --enc_in 7 \
      --d_model 128 \
      --dropout 0.5 \
      --des 'Exp' \
      --fix_seed "$seed"
  done
done