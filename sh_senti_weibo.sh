declare -a arr=("满意" "可以" "好" "行" "喜欢" "高兴" "开心")
# declare -a arr=("黑")

## now loop through the above array
for template in "${arr[@]}"
do
    echo "Run prompt zeroshot prediction for sentiment analysis"
    python run_prompt_zero_senti.py \
    --task weibo \
    --batch_size 32 \
    --model ./models/my_bert/ \
    --tokenizer ./models/my_bert/ \
    --output_dir ./senti_tuned_models \
    --result_output_dir ./zeroshot_results/ \
    --dataset_dir ./datasets/ \
    --label_convertion_flag False \
    --num_label_words 1 \
    --template=$template
done

# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"