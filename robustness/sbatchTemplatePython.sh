#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=8GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=10:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --account=student   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=axel.wong.2022@scis.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=robust_deepseek_test_1     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.9.21-GCCcore-13.3.0 
module load CUDA/11.8.0

# Create a virtual enviironment
python3 -m venv ~/myenv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/myenv/bin/activate

srun whichgpu

# If you require any packages, install it as usual before the srun job submission.
pip3 install numpy transformers torch tqdm pandas accelerate bitsandbytes

# Submit your job to the cluster
srun --gres=gpu:1 bash -c '
for model in "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"; do
    for task in "mnli" "qqp" "qnli" "rte"; do
        echo "Running model: $model, task: $task"
        python main.py --dataset advglue --task "$task" --model "$model" --service hug_gen --gpu 0
    done
    # Additional evaluation step
    # python main.py --dataset advglue_test --task "$task" --model "$model" --service hug_gen --gpu 0 --eval
done
'
