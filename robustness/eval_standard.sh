for task in  "sst2" "mnli" "rte" "qnli" "qqp";
    do
        python main.py --dataset advglue --task $task --service hug_gen --eval
    done