"python3 create_pretraining_data.py \
  --input_file=../bucket/dataset.txt \
  --output_file=../bucket/tf_examples.tfrecord \
  --vocab_file=../bucket/arabic-vocab.txt \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5"

python3 run_pretraining.py \
  --input_file=gs://arabert-bert/tf_examples.tfrecord \
  --output_dir=gs://arabert-bert/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://arabert-bert/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://arabert-bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=64 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=arabert-bert
