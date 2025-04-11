"""
Fairness Enhancement for DeepSeek-R1 Distill Qwen 7B

This script implements the fairness enhancement methods described in the project proposal:
1. Fair Prompt Optimization
2. Strategic Selection of Few-Shot Examples
3. Post-Processing Techniques

Usage:
    python main.py --input_file <path_to_bbq_data> --output_file <path_to_output> --method <enhancement_method>

Args:
    --input_file: Path to the BBQ dataset file
    --output_file: Path to save the enhanced results
    --method: Enhancement method to use (prompt_optimization, few_shot, post_processing, ensemble)
    --category: Bias category to focus on (e.g., Gender_identity, Race_ethnicity)
"""

import argparse
import json
import logging
import datetime
from pathlib import Path

from prompt_optimization import FairPromptOptimizer
from few_shot_selection import FewShotSelector
from post_processing import BiasCorrector
from ensemble import FairnessEnsemble
from model_wrapper import DeepSeekModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"fairness_enhancement_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fairness Enhancement for DeepSeek-R1")
    parser.add_argument('--input_file', type=str, required=True, 
                        help="Path to the BBQ dataset file")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the enhanced results")
    parser.add_argument('--method', type=str, required=True, 
                        choices=['prompt_optimization', 'few_shot', 'post_processing', 'ensemble'],
                        help="Enhancement method to use")
    parser.add_argument('--category', type=str, default=None,
                        help="Bias category to focus on (e.g., Gender_identity)")
    parser.add_argument('--meta_llm', type=str, default=None,
                        help="Meta LLM to use for prompt optimization")
    parser.add_argument('--num_examples', type=int, default=5,
                        help="Number of few-shot examples to use")
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path or identifier for the DeepSeek model")
    return parser.parse_args()

def load_data(file_path, category=None):
    """Load data from the BBQ dataset file."""
    logger.info(f"Loading data from {file_path}")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if category is None or item.get('category') == category:
                data.append(item)
    logger.info(f"Loaded {len(data)} examples")
    return data

def save_results(results, output_file):
    """Save enhanced results to output file."""
    logger.info(f"Saving results to {output_file}")
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    logger.info(f"Saved {len(results)} results")

def main():
    """Main function to run fairness enhancement."""
    args = parse_args()
    
    # Load data
    data = load_data(args.input_file, args.category)
    
    # Initialize model
    model = DeepSeekModel(args.model_path)
    
    # Run the appropriate enhancement method
    if args.method == 'prompt_optimization':
        logger.info("Using Fair Prompt Optimization method")
        enhancer = FairPromptOptimizer(model, meta_llm=args.meta_llm)
        results = enhancer.enhance(data)
    
    elif args.method == 'few_shot':
        logger.info("Using Strategic Few-Shot Selection method")
        enhancer = FewShotSelector(model, num_examples=args.num_examples)
        results = enhancer.enhance(data)
    
    elif args.method == 'post_processing':
        logger.info("Using Post-Processing method")
        enhancer = BiasCorrector(model)
        results = enhancer.enhance(data)
    
    elif args.method == 'ensemble':
        logger.info("Using Ensemble method")
        enhancer = FairnessEnsemble(model)
        results = enhancer.enhance(data)
    
    # Save results
    save_results(results, args.output_file)
    logger.info("Enhancement complete")

if __name__ == "__main__":
    main()
