"""
Run the prompt optimization method on a small sample of the BBQ dataset.

This script demonstrates the prompt optimization fairness enhancement method
with the DeepSeek-R1 model.
"""

import json
import os
import argparse
import logging
from tqdm import tqdm

from model_wrapper import DeepSeekModel
from prompt_optimization import FairPromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"fairness_enhancement/prompt_optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_sample_data(path, limit=5):
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

def run_prompt_optimization(input_path, output_file, sample_size=5):
    """Run prompt optimization on a sample of BBQ dataset."""
    # Load sample data
    data = load_sample_data(input_path, sample_size)
    if not data:
        logger.error("No data available to process")
        return
    
    logger.info(f"Loaded {len(data)} examples for processing")
    
    # Initialize DeepSeek model
    logger.info("Initializing DeepSeek model (this may take a moment)...")
    model = DeepSeekModel()
    
    # Initialize prompt optimizer
    logger.info("Initializing prompt optimizer...")
    enhancer = FairPromptOptimizer(model, iterations=1)  # Set to 1 iteration for demonstration
    
    # Apply the enhancement method
    try:
        logger.info(f"Applying prompt optimization to {len(data)} examples")
        enhanced_results = enhancer.enhance(data)
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in enhanced_results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Results saved to {output_file}")
        
        # Print some comparison stats
        ambiguous_examples = [ex for ex in enhanced_results if ex.get("context_condition") == "ambig"]
        if ambiguous_examples:
            unknown_before = 0
            unknown_after = 0
            
            for ex in ambiguous_examples:
                original_pred = ex.get("model_output", {}).get("original_prediction", "")
                enhanced_pred = ex.get("model_output", {}).get("prediction", "")
                
                options = [ex.get(f"ans{i}", "") for i in range(3)]
                unknown_option = options[2] if len(options) > 2 else None
                
                if unknown_option and "unknown" in unknown_option.lower():
                    if original_pred == unknown_option:
                        unknown_before += 1
                    if enhanced_pred == unknown_option:
                        unknown_after += 1
            
            total = len(ambiguous_examples)
            logger.info(f"Ambiguous contexts choosing 'unknown': Before: {unknown_before}/{total} ({unknown_before/total*100:.1f}%), After: {unknown_after}/{total} ({unknown_after/total*100:.1f}%)")
            logger.info(f"Improvement: {((unknown_after - unknown_before)/total*100):.1f}%")
            
    except Exception as e:
        logger.error(f"Error applying prompt optimization: {str(e)}")

def main():
    """Main function to parse arguments and run prompt optimization."""
    parser = argparse.ArgumentParser(description="Run prompt optimization on BBQ dataset")
    parser.add_argument('--input', type=str, default="fairness/Bias-Benchmark/data/Gender_identity.jsonl",
                       help="Path to BBQ dataset file")
    parser.add_argument('--output', type=str, default="fairness_enhancement/results/prompt_optimization_results.jsonl",
                       help="Path to save results")
    parser.add_argument('--sample_size', type=int, default=5,
                       help="Number of examples to process (default: 5)")
    
    args = parser.parse_args()
    
    # Run prompt optimization
    run_prompt_optimization(args.input, args.output, args.sample_size)

if __name__ == "__main__":
    main()
