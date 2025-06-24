export CUDA_VISIBLE_DEVICES=1

model_name="PatchTST"
seed=${1:-123}

for optimizer in "TS_Adam" "Adam" "SGD" "Yogi" "AdamW" "lookahead_Adam" ; do
  for seasonal_patterns in 'Hourly' ; do
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
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model 32 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --fix_seed "$seed" \
      --batch_size 8 \
      --train_epochs 20
  done
done