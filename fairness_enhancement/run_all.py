""" Run all fairness enhancement methods across all categories. """

import argparse
import json
import logging
import os
import random
import subprocess


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


def run_method(category, method, input_file):
    """Run a single fairness enhancement method and handle errors properly."""
    output_file = f"results/{category}_{method}_results.jsonl"
    cmd = f"python main.py --input_file {input_file} --output_file {output_file} --method {method} --category {category}"
    if method == "few_shot":
        cmd += " --num_examples 5"

    try:
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        logger.warning(f"Skipping to next method due to error")
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
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/samples", exist_ok=True)

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
            run_method(category, method, input_file)

    # Run analysis on results
    logger.info("Running analysis on all results")
    subprocess.run("python analyze_results.py", shell=True, check=False)

    logger.info("All fairness enhancement methods completed!")


if __name__ == "__main__":
    main()
