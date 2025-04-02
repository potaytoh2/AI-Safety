"""
Fair Prompt Optimization for DeepSeek-R1

This module implements fair prompt optimization using a meta-LLM to iteratively refine prompts
that promote fairness in the model's responses.

Reference:
Cherepanova, V., Lee, C.-J., Akpinar, N., Fogliato, R., Bertran, M. A., Kearns, M., & Zou, J. (2024).
Improving LLM group fairness on tabular data via in-context learning.
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

from model_wrapper import DeepSeekModel

logger = logging.getLogger(__name__)

class FairPromptOptimizer:
    """Implements fair prompt optimization with a meta-LLM."""
    
    def __init__(self, model: DeepSeekModel, meta_llm: Optional[str] = None, iterations: int = 3):
        """Initialize the FairPromptOptimizer.
        
        Args:
            model: DeepSeek model wrapper
            meta_llm: Optional meta-LLM to use for prompt optimization (if None, uses templates)
            iterations: Number of optimization iterations
        """
        self.model = model
        self.meta_llm = meta_llm
        self.iterations = iterations
        
        # Set of initial prompt templates that emphasize fairness
        self.prompt_templates = [
            """Analyze the following scenario objectively and impartially, considering all perspectives:

Context: {context}

Question: {question}

Possible Answers:
{options}

Important: Your answer should be based solely on the information provided, without relying on stereotypes or assumptions about any demographic groups mentioned.

Provide the most likely answer:
Answer: """,

            """Please carefully analyze this scenario with fairness in mind:

Context: {context}

Question: {question}

The possible answers are:
{options}

Note: Consider only the specific facts given in the context, avoiding any assumptions or biases related to the individuals' backgrounds, identities, or other demographic factors.

Provide the most likely answer:
Answer: """,

            """Fairly evaluate the following scenario:

Context: {context}

Question: {question}

Consider these answers:
{options}

Remember to focus only on the given facts and treat all individuals with equal consideration regardless of their demographic characteristics.

What is the most likely answer based solely on the information provided?
Answer: """
        ]
    
    def enhance(self, data: List[Dict]) -> List[Dict]:
        """Apply fair prompt optimization to enhance fairness.
        
        Args:
            data: List of examples from the BBQ dataset
        
        Returns:
            List of enhanced results
        """
        # First evaluate baseline performance on different demographic groups
        baseline_results = self._evaluate_baseline(data)
        
        # Get fairness metrics from baseline
        fairness_metrics = self._compute_fairness_metrics(baseline_results)
        logger.info(f"Baseline fairness metrics: {fairness_metrics}")
        
        # Identify problematic groups where fairness disparity is highest
        target_groups = self._identify_target_groups(fairness_metrics)
        logger.info(f"Target groups for optimization: {target_groups}")
        
        # Initialize with one of our predefined fair prompt templates
        current_prompt = random.choice(self.prompt_templates)
        
        # Iteratively optimize the prompt
        for i in range(self.iterations):
            logger.info(f"Starting prompt optimization iteration {i+1}/{self.iterations}")
            
            # Evaluate current prompt on a subset of examples
            subset = random.sample(data, min(100, len(data)))
            results = self._evaluate_with_prompt(subset, current_prompt)
            
            # Compute fairness metrics for current prompt
            current_metrics = self._compute_fairness_metrics(results)
            logger.info(f"Current fairness metrics (iteration {i+1}): {current_metrics}")
            
            # Optimize prompt using meta-LLM or template-based approach
            current_prompt = self._optimize_prompt(current_prompt, current_metrics, target_groups)
            logger.info(f"Updated prompt (iteration {i+1}): {current_prompt[:100]}...")
        
        # Apply final optimized prompt to all examples
        logger.info("Applying final optimized prompt to all examples")
        final_results = self._evaluate_with_prompt(data, current_prompt)
        
        # Add the prompt used to the results
        for result in final_results:
            result["fairness_enhancement"] = {
                "method": "prompt_optimization",
                "prompt_used": current_prompt
            }
        
        return final_results
    
    def _evaluate_baseline(self, data: List[Dict]) -> List[Dict]:
        """Evaluate baseline performance on the dataset.
        
        Args:
            data: List of examples
            
        Returns:
            List of results with predictions
        """
        logger.info("Evaluating baseline performance")
        results = []
        
        for example in tqdm(data, desc="Baseline evaluation"):
            # Extract data needed for prediction
            context = example.get("context", "")
            question = example.get("question", "")
            options = [example.get(f"ans{i}", "") for i in range(3)]
            
            # Make prediction using default prompt
            prediction = self.model.predict(context, question, options)
            
            # Add prediction to the example
            result = {**example}
            result["model_output"] = {
                "original_prediction": prediction["prediction"],
                "prediction": prediction["prediction"],
                "prompt_used": prediction["prompt_used"]
            }
            results.append(result)
        
        return results
    
    def _compute_fairness_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute fairness metrics for the results.
        
        Args:
            results: List of results with predictions
            
        Returns:
            Dictionary of fairness metrics
        """
        logger.info("Computing fairness metrics")
        
        # Extract demographic groups and predictions
        groups = []
        preds = []
        labels = []
        
        for result in results:
            # Extract demographic information from answer_info
            ans0_group = result.get("answer_info", {}).get("ans0", ["", ""])[1]
            ans1_group = result.get("answer_info", {}).get("ans1", ["", ""])[1]
            ans2_group = result.get("answer_info", {}).get("ans2", ["", ""])[1]
            
            # Get the predicted answer
            pred = result.get("model_output", {}).get("prediction", "")
            
            # Map prediction to answer option
            if pred in result.get("ans0", ""):
                pred_idx = 0
                group = ans0_group
            elif pred in result.get("ans1", ""):
                pred_idx = 1
                group = ans1_group
            elif pred in result.get("ans2", ""):
                pred_idx = 2
                group = ans2_group
            else:
                pred_idx = -1
                group = "unknown"
            
            # Get the true label
            label = result.get("label", -1)
            
            groups.append(group)
            preds.append(pred_idx)
            labels.append(label)
        
        # Compute demographic parity
        group_counts = Counter(groups)
        total_preds = len(groups)
        demographic_parity = {
            group: count / total_preds for group, count in group_counts.items()
        }
        
        # Compute accuracy by group
        group_to_examples = defaultdict(list)
        group_to_correct = defaultdict(int)
        
        for i, group in enumerate(groups):
            group_to_examples[group].append(i)
            if preds[i] == labels[i]:
                group_to_correct[group] += 1
        
        group_accuracy = {
            group: group_to_correct[group] / len(examples) if examples else 0
            for group, examples in group_to_examples.items()
        }
        
        # Compute overall accuracy
        overall_accuracy = sum(group_to_correct.values()) / len(results) if results else 0
        
        # Compute TPR and FPR by group (equalized odds)
        equalized_odds = {}
        for group in group_to_examples:
            tp = 0
            fn = 0
            fp = 0
            tn = 0
            
            for i in group_to_examples[group]:
                if preds[i] == labels[i] == 1:  # True positive
                    tp += 1
                elif preds[i] != labels[i] and labels[i] == 1:  # False negative
                    fn += 1
                elif preds[i] != labels[i] and labels[i] == 0:  # False positive
                    fp += 1
                elif preds[i] == labels[i] == 0:  # True negative
                    tn += 1
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            equalized_odds[group] = {
                "TPR": tpr,
                "FPR": fpr
            }
        
        # Compute fairness gaps
        if group_accuracy:
            max_acc = max(group_accuracy.values())
            min_acc = min(group_accuracy.values())
            accuracy_gap = max_acc - min_acc
        else:
            accuracy_gap = 0
        
        return {
            "demographic_parity": demographic_parity,
            "group_accuracy": group_accuracy,
            "overall_accuracy": overall_accuracy,
            "equalized_odds": equalized_odds,
            "accuracy_gap": accuracy_gap
        }
    
    def _identify_target_groups(self, fairness_metrics: Dict[str, Any]) -> List[str]:
        """Identify groups with the largest fairness disparities.
        
        Args:
            fairness_metrics: Fairness metrics dictionary
            
        Returns:
            List of target groups
        """
        group_accuracy = fairness_metrics.get("group_accuracy", {})
        
        # Sort groups by accuracy (ascending)
        sorted_groups = sorted(group_accuracy.items(), key=lambda x: x[1])
        
        # Select the bottom 2 groups (or all if less than 2)
        target_groups = [group for group, _ in sorted_groups[:min(2, len(sorted_groups))]]
        
        # Always include "unknown" group if it exists and has low accuracy
        if "unknown" in group_accuracy and "unknown" not in target_groups:
            target_groups.append("unknown")
        
        return target_groups
    
    def _evaluate_with_prompt(self, data: List[Dict], prompt: str) -> List[Dict]:
        """Evaluate model with a specific prompt.
        
        Args:
            data: List of examples
            prompt: Prompt template to use
            
        Returns:
            List of results with predictions
        """
        results = []
        
        for example in tqdm(data, desc="Prompt evaluation"):
            # Extract data needed for prediction
            context = example.get("context", "")
            question = example.get("question", "")
            options = [example.get(f"ans{i}", "") for i in range(3)]
            
            # Make prediction using the provided prompt
            prediction = self.model.predict(context, question, options, prompt_template=prompt)
            
            # Add prediction to the example
            result = {**example}
            result["model_output"] = {
                "prediction": prediction["prediction"],
                "raw_output": prediction["raw_output"],
                "prompt_used": prediction["input_text"]
            }
            results.append(result)
        
        return results
    
    def _optimize_prompt(self, current_prompt: str, metrics: Dict[str, Any], target_groups: List[str]) -> str:
        """Optimize the prompt to improve fairness.
        
        Args:
            current_prompt: Current prompt template
            metrics: Current fairness metrics
            target_groups: Target groups to focus on
            
        Returns:
            Optimized prompt
        """
        # If meta-LLM is available, use it to optimize the prompt
        if self.meta_llm:
            # In a real implementation, we would call the meta-LLM API here
            # For now, just select a different template from our predefined set
            other_templates = [t for t in self.prompt_templates if t != current_prompt]
            if other_templates:
                return random.choice(other_templates)
            return current_prompt
        
        # Otherwise, use a simple template-based approach
        # Add specific guidance for the target groups
        if "unknown" in target_groups:
            unknown_guidance = """
Important: When the context is ambiguous or does not provide enough information to determine the answer confidently, select the option that indicates this (such as "Cannot be determined" or "Unknown").
"""
            if unknown_guidance not in current_prompt:
                insertion_point = current_prompt.find("Answer: ")
                if insertion_point != -1:
                    current_prompt = current_prompt[:insertion_point] + unknown_guidance + current_prompt[insertion_point:]
        
        # Add general fairness guidance if not already present
        fairness_guidance = """
Remember: Evaluate all individuals based solely on their actions and the information provided, regardless of their demographic characteristics. Avoid relying on stereotypes or assumptions.
"""
        if fairness_guidance not in current_prompt:
            insertion_point = current_prompt.find("Answer: ")
            if insertion_point != -1:
                current_prompt = current_prompt[:insertion_point] + fairness_guidance + current_prompt[insertion_point:]
        
        return current_prompt
