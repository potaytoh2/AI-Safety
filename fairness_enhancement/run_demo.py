"""
Demonstration script for fairness enhancement methods.

This script shows how to run the fairness enhancement methods with the real
DeepSeek model on BBQ dataset samples.
"""

import json
import os
import argparse
import logging
from tqdm import tqdm

# Import fairness enhancement methods
from model_wrapper import DeepSeekModel
from prompt_optimization import FairPromptOptimizer
from few_shot_selection import FewShotSelector
from post_processing import BiasCorrector
from ensemble import FairnessEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"fairness_enhancement/fairness_enhancement.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_sample_data(path, limit=10):
    """Load a small sample of data from the BBQ dataset."""
    logger.info(f"Loading sample data from {path}")
    data = []
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        return []

def run_fairness_enhancement(input_path, output_dir, sample_size=10):
    """Run fairness enhancement on BBQ dataset."""
    # Load sample data
    data = load_sample_data(input_path, sample_size)
    if not data:
        logger.error("No data available to process")
        return
    
    logger.info(f"Loaded {len(data)} examples for processing")
    
    # Initialize DeepSeek model
    logger.info("Initializing DeepSeek model (this may take a moment)...")
    model = DeepSeekModel()
    
    # Process with different enhancement methods
    methods = ["prompt_optimization", "few_shot", "post_processing", "ensemble"]
    
    for method in methods:
        logger.info(f"Running {method} method")
        
        # Initialize the appropriate enhancer based on the method
        if method == "prompt_optimization":
            enhancer = FairPromptOptimizer(model)
        elif method == "few_shot":
            enhancer = FewShotSelector(model)
        elif method == "post_processing":
            enhancer = BiasCorrector(model)
        elif method == "ensemble":
            enhancer = FairnessEnsemble(model)
        else:
            logger.error(f"Unknown method: {method}")
            continue
        
        # Apply the enhancement method
        try:
            logger.info(f"Applying {method} to {len(data)} examples")
            enhanced_results = enhancer.enhance(data)
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{method}_results.jsonl")
            
            with open(output_path, 'w') as f:
                for result in enhanced_results:
                    f.write(json.dumps(result) + '\n')
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error applying {method}: {str(e)}")

def main():
    """Main function to parse arguments and run fairness enhancement."""
    parser = argparse.ArgumentParser(description="Run fairness enhancement on BBQ dataset")
    parser.add_argument('--input', type=str, default="fairness/Bias-Benchmark/data/Gender_identity.jsonl",
                       help="Path to BBQ dataset file")
    parser.add_argument('--output', type=str, default="fairness_enhancement/results",
                       help="Directory to save results")
    parser.add_argument('--sample_size', type=int, default=10,
                       help="Number of examples to process (default: 10)")
    
    args = parser.parse_args()
    
    # Run fairness enhancement
    run_fairness_enhancement(args.input, args.output, args.sample_size)

if __name__ == "__main__":
    main()
