#!/bin/bash

#################################################
## FAIRNESS ENHANCEMENT BATCH JOB             ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=16GB                  # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=24:00:00           # How long to run the job for? Format <DD-HH:MM:SS>
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --account=student           # The account you've been assigned (normally student)
#SBATCH --qos=studentqos            # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=qijie.lim.2022@scis.smu.edu.sg  # Your email for notifications
#SBATCH --job-name=fairness_run_all  # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require
module purge
module load Python/3.9.21-GCCcore-13.3.0 
module load CUDA/11.8.0

# Create a virtual environment (or use an existing one)
VENV_DIR=~/fairness_venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Print GPU information
srun whichgpu

# Install required packages (if not already installed)
pip install -q transformers torch tqdm pandas accelerate bitsandbytes

# Navigate to the correct directory
# The script is already in fairness_enhancement, so we need to make sure we're in that directory
echo "Navigating to the correct directory..."
cd "$SLURM_SUBMIT_DIR" || exit 1  # Go to submission directory
if [[ $(basename "$(pwd)") != "fairness_enhancement" ]]; then
    echo "Not in fairness_enhancement directory, navigating there..."
    cd fairness_enhancement || exit 1
fi
echo "Current directory: $(pwd)"

# Install project-specific requirements
pip install -q -r requirements.txt

# Run the specific Race_x_gender ensemble model first
echo "Running specific Race_x_gender ensemble task..."
srun --gres=gpu:1 python main.py \
    --input_file results/samples/Race_x_gender_sample_100.jsonl \
    --output_file results/Race_x_gender_ensemble_results.jsonl \
    --method ensemble \
    --category Race_x_gender

# Define categories and methods
echo "Setting up categories and methods..."
CATEGORIES=(
    "Race_x_SES"
    "Religion"
    "SES"
    "Sexual_orientation"
)

METHODS=(
    "prompt_optimization"
    "few_shot"
    "post_processing"
    "ensemble"
)

# Ensure results/samples directory exists
mkdir -p results/samples

# Run each category and method individually
for category in "${CATEGORIES[@]}"; do
    echo "=== Processing category: $category ==="
    
    # Create sample files if needed
    input_file="../fairness/Bias-Benchmark/data/${category}.jsonl"
    sample_file="results/samples/${category}_sample_100.jsonl"
    
    if [ ! -f "$sample_file" ] && [ -f "$input_file" ]; then
        echo "Creating sample for $category (100 examples)..."
        python -c "
import json, random
with open('$input_file', 'r') as f:
    lines = f.readlines()
sample = random.sample(lines, min(100, len(lines)))
with open('$sample_file', 'w') as f:
    f.writelines(sample)
print(f'Created sample with {len(sample)} examples')
"
    fi
    
    # Run each method for this category
    for method in "${METHODS[@]}"; do
        output_file="results/${category}_${method}_results.jsonl"
        
        echo "Running $method for $category..."
        
        # Add num_examples parameter for few_shot method
        if [ "$method" == "few_shot" ]; then
            extra_args="--num_examples 5"
        else
            extra_args=""
        fi
        
        # Run the command
        srun --gres=gpu:1 python main.py \
            --input_file "$sample_file" \
            --output_file "$output_file" \
            --method "$method" \
            --category "$category" \
            $extra_args
        
        echo "Completed $method for $category"
    done
done

# Run analysis
echo "Running analysis on results..."
srun python analyze_results.py

echo "Job completed!"
