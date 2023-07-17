for split in 0 1 2 3 4
do
    echo "Run prompt fewshot train and prediction for sentiment analysis"
    python run_prompt_few_senti_same_epochs_v1.py \
    --task ecommerce \
    --batch_size 2 \
    --eval_batch_size 64 \
    --lr 1e-4 \
    --epochs 20 \
    --model ./models/my_bert/ \
    --tokenizer ./models/my_bert/ \
    --output_dir ./tuned_models_eq \
    --result_output_dir ./fewshot_results_eq/ \
    --dataset_dir ./datasets/ \
    --label_convertion_flag False \
    --max_len 512 \
    --num_label_words 1 \
    --seed $split
done