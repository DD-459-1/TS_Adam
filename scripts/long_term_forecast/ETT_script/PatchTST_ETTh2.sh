export CUDA_VISIBLE_DEVICES=0

model_name="PatchTST"
seed=${1:-123}


for pred_len in 96 192 336 720 ; do
  for optimizer in "AutoCyclic" ; do
    python -u run.py \
      --optimizer "$optimizer" \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "./dataset/ETT-small/" \
      --data_path "ETTh2.csv" \
      --model_id "ETTh2_96_$pred_len" \
      --model "$model_name" \
      --data "ETTh2" \
      --features "M" \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --e_layers 1 \
      --d_layers 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --batch_size 64 \
      --des "Exp" \
      --d_model 64 \
      --d_ff 64 \
      --itr 1 \
      --fix_seed "$seed"
  done
done