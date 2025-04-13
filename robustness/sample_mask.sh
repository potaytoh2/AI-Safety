for mask in 0.05 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for task in "sst2" 
    do
        echo $model $task
        python main.py --dataset advglue --task $task --model "gemini-1.5-flash" --service hug_gen --mask_rate $mask
    done
    #python main.py --dataset advglue --task "sst2" --model $model --service hug_gen --eval --mask_rate 0.1
done