export CUDA_VISIBLE_DEVICES=0

model_name="MICN"
seed=${1:-123}
batch_size=${2:-32}

for pred_len in 96 192 336 720; do
  for optimizer in "TS_Adam" "Adam" "SGD" "Yogi" "AdamW" "lookahead_Adam" ; do
    python -u run.py \
      --optimizer "$optimizer" \
      --batch_size "$batch_size" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "./dataset/ETT-small/" \
      --data_path "ETTh2.csv" \
      --model_id "ETTh2_96_$pred_len" \
      --model "$model_name" \
      --data "ETTh2" \
      --features "M" \
      --seq_len 96 \
      --label_len 96 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "Exp" \
      --itr 1 \
      --fix_seed "$seed"
  done
done