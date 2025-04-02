for model in "gemini-1.5-flash" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"gemini-1.5-flash-8b"
do
    for task in "sst2" "mnli" "qqp" "rte" "qnli"
    do
        echo $model $task
        python main.py --dataset advglue --task $task --model $model --service hug_gen --mask_rate 0.1
    done
    #python main.py --dataset advglue --task "sst2" --model $model --service hug_gen --eval
done