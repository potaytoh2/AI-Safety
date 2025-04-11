""" Run all fairness enhancement methods across all categories with improved efficiency. """

import argparse
import json
import logging
import os
import random
import subprocess
import time
import sys
from pathlib import Path
from tqdm import tqdm

# Add the current directory to the path so we can import our modules
sys.path.append(os.getcwd())

# Import necessary modules for direct model reuse
try:
    from model_wrapper import DeepSeekModel
    from prompt_optimization import FairPromptOptimizer
    from few_shot_selection import FewShotSelector
    from post_processing import BiasCorrector
    from ensemble import FairnessEnsemble
    MODEL_IMPORT_SUCCESS = True
except ImportError as e:
    logger.warning(f"Could not import model modules: {e}")
    MODEL_IMPORT_SUCCESS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define categories from BBQ benchmark
CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation"
]

# Define methods
METHODS = [
    "prompt_optimization",
    "few_shot",
    "post_processing",
    "ensemble"
]


def run_method_external(category, method, input_file):
    """Run a single fairness enhancement method using subprocess and show progress."""
    output_file = f"results/{category}_{method}_results.jsonl"
    cmd = f"python main.py --input_file {input_file} --output_file {output_file} --method {method} --category {category}"
    if method == "few_shot":
        cmd += " --num_examples 5"

    try:
        logger.info(f"Running: {cmd}")
        
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Set up progress bar for this specific command
        pbar = tqdm(desc=f"{category} - {method}", leave=True)
        
        # Process output as it comes
        while True:
            # Read one line of output
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                # Remove trailing newline
                line = line.rstrip()
                
                # Update progress bar description with status
                if "INFO" in line:
                    pbar.set_description(f"{category} - {method}: {line.split('INFO - ')[-1]}")
                    
                    # Update progress bar based on certain keywords
                    if "Baseline evaluation" in line:
                        pbar.update(10)
                    elif "Computing fairness metrics" in line:
                        pbar.update(15)
                    elif "prompt optimization iteration" in line:
                        pbar.update(20)
                    elif "Applying final optimized prompt" in line:
                        pbar.update(25)
                    elif "Processing example" in line or "Processing batch" in line:
                        pbar.update(1)
                    elif "few-shot examples" in line:
                        pbar.update(10)
                    elif "ensemble prediction" in line:
                        pbar.update(5)
                
                # Also print the output for logging
                logger.info(line)
        
        # Check for errors in stderr
        stderr = process.stderr.read()
        if stderr:
            logger.error(f"Error output: {stderr}")
        
        # Close the progress bar
        pbar.close()
        
        # Check return code
        if process.returncode != 0:
            logger.error(f"Command failed with code {process.returncode}")
            return False
            
        logger.info("Command completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        logger.warning(f"Skipping to next method due to error")
        return False


def run_method_direct(model, category, method, input_file, output_file, num_examples=5):
    """Run a single fairness enhancement method directly using the loaded model with progress tracking."""
    try:
        logger.info(f"Running {method} on {category} using direct model call")
        
        # Set up progress bar
        pbar = tqdm(desc=f"{category} - {method}", leave=True)
        
        # Load data with progress update
        pbar.set_description(f"{category} - {method}: Loading data from {input_file}")
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        pbar.update(10)
        logger.info(f"Loaded {len(data)} examples from {input_file}")
        
        # Create the appropriate enhancer
        if method == 'prompt_optimization':
            pbar.set_description(f"{category} - {method}: Initializing prompt optimizer")
            enhancer = FairPromptOptimizer(model)
            pbar.update(5)
        elif method == 'few_shot':
            pbar.set_description(f"{category} - {method}: Initializing few-shot selector")
            enhancer = FewShotSelector(model, num_examples=num_examples)
            pbar.update(5)
        elif method == 'post_processing':
            pbar.set_description(f"{category} - {method}: Initializing bias corrector")
            enhancer = BiasCorrector(model)
            pbar.update(5)
        elif method == 'ensemble':
            pbar.set_description(f"{category} - {method}: Initializing fairness ensemble")
            enhancer = FairnessEnsemble(model)
            pbar.update(5)
        
        # Custom progress callback for the enhancer
        def progress_callback(status, progress):
            pbar.set_description(f"{category} - {method}: {status}")
            pbar.update(progress)
        
        # Add progress callback to enhancer if it has an attribute for it
        if hasattr(enhancer, 'set_progress_callback'):
            enhancer.set_progress_callback(progress_callback)
        
        # Run enhancement
        pbar.set_description(f"{category} - {method}: Running enhancement")
        results = enhancer.enhance(data)
        pbar.update(50)
        
        # Save results
        pbar.set_description(f"{category} - {method}: Saving results to {output_file}")
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        pbar.update(10)
        pbar.set_description(f"{category} - {method}: Completed successfully")
        
        # Close the progress bar
        pbar.close()
        logger.info(f"Saved {len(results)} results to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error in direct method execution: {str(e)}")
        return False


def sample_data(input_file, output_file, sample_size):
    """Sample data from an input file and write to an output file."""
    try:
        # Read all lines from the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Sample random lines
        if sample_size >= len(lines):
            sampled_lines = lines
        else:
            sampled_lines = random.sample(lines, sample_size)
        
        # Write sampled lines to the output file
        with open(output_file, 'w') as f:
            f.writelines(sampled_lines)
        
        logger.info(f"Sampled {len(sampled_lines)} out of {len(lines)} examples")
        return True
    except Exception as e:
        logger.error(f"Error sampling data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all fairness enhancement methods')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of examples to sample (default: use all data)')
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                        help='Path to the model to use')
    parser.add_argument('--use_direct', action='store_true', 
                        help='Use direct model loading for better efficiency (requires more memory)')
    parser.add_argument('--retry_count', type=int, default=1, 
                        help='Number of times to retry failed methods')
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/samples", exist_ok=True)

    # Calculate total number of tasks
    total_tasks = len(CATEGORIES) * len(METHODS)
    
    # Initialize model if using direct mode
    model = None
    if args.use_direct and MODEL_IMPORT_SUCCESS:
        try:
            logger.info(f"Initializing model from {args.model_path} for reuse")
            model = DeepSeekModel(args.model_path)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.warning("Falling back to subprocess mode")
            args.use_direct = False
    elif args.use_direct:
        logger.warning("Direct mode requested but model modules could not be imported")
        logger.warning("Falling back to subprocess mode")
        args.use_direct = False
    
    # Create a master progress bar
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        # Run each method on each category
        for category in CATEGORIES:
            logger.info(f"Processing category: {category}")
            
            input_file = f"../fairness/Bias-Benchmark/data/{category}.jsonl"
            
            # If sampling is requested, create a sampled version of the data
            if args.sample_size:
                sample_file = f"results/samples/{category}_sample_{args.sample_size}.jsonl"
                if sample_data(input_file, sample_file, args.sample_size):
                    input_file = sample_file
                else:
                    logger.error(f"Failed to create sample for {category}, skipping")
                    continue
            
            for method in METHODS:
                output_file = f"results/{category}_{method}_results.jsonl"
                
                # Try to run the method with retries if needed
                success = False
                for attempt in range(args.retry_count):
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt} for {category} - {method}")
                    
                    if args.use_direct and model is not None:
                        # Run using direct model access
                        success = run_method_direct(
                            model, 
                            category, 
                            method, 
                            input_file, 
                            output_file,
                            num_examples=5 if method == "few_shot" else None
                        )
                    else:
                        # Run using subprocess
                        success = run_method_external(category, method, input_file)
                    
                    if success:
                        break
                
                # Update progress bar
                pbar.update(1)
                status = "Completed" if success else "Failed"
                pbar.set_description(f"{status}: {category} - {method}")

    # Run analysis on results
    logger.info("Running analysis on all results")
    subprocess.run("python analyze_results.py", shell=True, check=False)

    logger.info("All fairness enhancement methods completed!")


if __name__ == "__main__":
    main()
