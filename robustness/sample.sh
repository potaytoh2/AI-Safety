for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
do
    for task in "sst2"
    do
        echo $model $task
        python main.py --dataset advglue_test --task $task --model $model --service hug_gen
    done
    python main.py --dataset advglue_test --task "sst2" --model $model --service hug_gen --eval
done