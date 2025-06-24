export CUDA_VISIBLE_DEVICES=0

model_name="MICN"
seed=${1:-123}


for optimizer in  "AutoCyclic" ; do
  for pred_len in 96 192 336 720; do
    python -u run.py \
      --optimizer "$optimizer" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "./dataset/ETT-small/" \
      --data_path "ETTm1.csv" \
      --model_id "ETTm1_96_$pred_len" \
      --model "$model_name" \
      --data "ETTm1" \
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
      --top_k 5 \
      --itr 1 \
      --fix_seed "$seed"
  done
done