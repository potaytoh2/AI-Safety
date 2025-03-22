for model in "gemini-1.5-flash-8b" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #"gemini-1.5-flash-8b"
do
    for task in "mnli" "qqp" #"rte"
    do
        echo $model $task
        python main.py --dataset advglue --task $task --model $model --service hug_gen
    done
    #python main.py --dataset advglue --task "sst2" --model $model --service hug_gen --eval
done