"""
Strategic Few-Shot Selection for DeepSeek-R1

This module implements strategic few-shot example selection to improve fairness in the model's responses
by carefully curating diverse and balanced examples for in-context learning.

Reference:
Hu, J., Liu, W., & Du, M. (2024). Strategic demonstration selection for improved fairness in LLM in-context learning.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np

from model_wrapper import DeepSeekModel

logger = logging.getLogger(__name__)

class FewShotSelector:
    """Implements strategic few-shot selection for fairness enhancement."""
    
    def __init__(self, model: DeepSeekModel, num_examples: int = 5, similarity_threshold: float = 0.7):
        """Initialize the FewShotSelector.
        
        Args:
            model: DeepSeek model wrapper
            num_examples: Number of few-shot examples to include
            similarity_threshold: Threshold for semantic similarity when selecting diverse examples
        """
        self.model = model
        self.num_examples = num_examples
        self.similarity_threshold = similarity_threshold
        
        # Base prompt template
        self.base_prompt = """Analyze the scenario:

Context: {context}

Question: {question}

Possible Answers:
{options}

Provide the most likely answer:
Answer: """
    
    def enhance(self, data: List[Dict]) -> List[Dict]:
        """Apply strategic few-shot selection to enhance fairness.
        
        Args:
            data: List of examples from the BBQ dataset
        
        Returns:
            List of enhanced results with few-shot examples
        """
        logger.info("Applying strategic few-shot selection")
        
        # First, identify the demographic groups in the dataset
        groups = self._identify_groups(data)
        logger.info(f"Identified demographic groups: {groups}")
        
        # Create a pool of examples for each demographic group
        group_pools = self._create_group_pools(data, groups)
        
        # Identify ambiguous examples where the model might show bias
        ambiguous_examples = [ex for ex in data if ex.get("context_condition") == "ambig"]
        logger.info(f"Identified {len(ambiguous_examples)} ambiguous examples")
        
        # Evaluate model on ambiguous examples to find challenging cases
        challenging_examples = self._identify_challenging_examples(ambiguous_examples)
        logger.info(f"Identified {len(challenging_examples)} challenging examples")
        
        # Process all examples with strategically selected few-shot examples
        results = []
        for example in tqdm(data, desc="Processing with few-shot examples"):
            try:
                # Select few-shot examples based on the current example
                few_shot_examples = self._select_few_shot_examples(example, group_pools, challenging_examples)
                
                # Extract data needed for prediction
                context = example.get("context", "")
                question = example.get("question", "")
                options = [example.get(f"ans{i}", "") for i in range(3)]
                
                # Make prediction with few-shot examples
                prediction = self.model.predict(context, question, options, 
                                               prompt_template=self.base_prompt,
                                               few_shot_examples=few_shot_examples)
                
                # Add prediction and few-shot examples to the result
                result = {**example}
                result["model_output"] = {
                    "prediction": prediction.get("prediction", ""),
                    "prompt_used": prediction.get("input_text", "")
                }
                
                # Only add raw_output if it exists
                if isinstance(prediction, dict) and "raw_output" in prediction:
                    result["model_output"]["raw_output"] = prediction["raw_output"]
                
                result["fairness_enhancement"] = {
                    "method": "few_shot_selection",
                    "few_shot_examples": [ex["example_id"] for ex in few_shot_examples]
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing example: {str(e)}")
                # Add a minimal result to maintain consistency
                results.append({
                    **example,
                    "model_output": {"error": str(e)},
                    "fairness_enhancement": {"method": "few_shot_selection", "error": True}
                })
        
        return results
    
    def _identify_groups(self, data: List[Dict]) -> List[str]:
        """Identify demographic groups in the dataset.
        
        Args:
            data: List of examples
            
        Returns:
            List of demographic groups
        """
        # Extract all groups from answer_info
        all_groups = set()
        for example in data:
            answer_info = example.get("answer_info", {})
            for i in range(3):
                ans_key = f"ans{i}"
                if ans_key in answer_info:
                    group = answer_info[ans_key][1] if len(answer_info[ans_key]) > 1 else "unknown"
                    if group:
                        all_groups.add(group)
        
        return list(all_groups)
    
    def _create_group_pools(self, data: List[Dict], groups: List[str]) -> Dict[str, List[Dict]]:
        """Create pools of examples for each demographic group.
        
        Args:
            data: List of examples
            groups: List of demographic groups
            
        Returns:
            Dictionary mapping groups to lists of examples
        """
        # Initialize pools
        pools = {group: [] for group in groups}
        
        # Add examples to appropriate pools
        for example in data:
            answer_info = example.get("answer_info", {})
            label = example.get("label", -1)
            
            # Only add disambiguated examples with correct answers as few-shot examples
            if example.get("context_condition") == "disambig" and label != -1:
                ans_key = f"ans{label}"
                if ans_key in answer_info:
                    group = answer_info[ans_key][1] if len(answer_info[ans_key]) > 1 else "unknown"
                    if group in pools:
                        pools[group].append(example)
        
        # Ensure each pool has at least a minimum number of examples
        for group, examples in pools.items():
            if len(examples) < 2:  # Need at least a few examples per group
                logger.warning(f"Group {group} has only {len(examples)} examples. Adding examples from other groups.")
                # Add examples from other groups if needed
                needed = max(2 - len(examples), 0)
                other_examples = []
                for other_group, other_pool in pools.items():
                    if other_group != group and len(other_pool) > 2:
                        other_examples.extend(other_pool[:2])
                
                if other_examples:
                    random.shuffle(other_examples)
                    pools[group].extend(other_examples[:needed])
        
        return pools
    
    def _identify_challenging_examples(self, ambiguous_examples: List[Dict]) -> List[Dict]:
        """Identify challenging examples where the model might show bias.
        
        Args:
            ambiguous_examples: List of ambiguous examples
            
        Returns:
            List of challenging examples
        """
        # If there are too many ambiguous examples, sample a subset
        if len(ambiguous_examples) > 50:
            sample_size = min(50, len(ambiguous_examples))
            sampled_examples = random.sample(ambiguous_examples, sample_size)
        else:
            sampled_examples = ambiguous_examples
        
        # Evaluate model on sampled examples
        challenging = []
        for example in tqdm(sampled_examples, desc="Identifying challenging examples"):
            try:
                context = example.get("context", "")
                question = example.get("question", "")
                options = [example.get(f"ans{i}", "") for i in range(3)]
                
                # Make prediction using default prompt
                prediction = self.model.predict(context, question, options)
                
                # Extract the predicted answer and check if it matches "unknown"
                predicted = prediction.get("prediction", "")
                
                # If the model predicts a specific answer for an ambiguous context,
                # it might be relying on stereotypes or biases
                if "unknown" not in predicted.lower():
                    challenging.append(example)
            except Exception as e:
                logger.error(f"Error evaluating challenging example: {str(e)}")
        
        return challenging
    
    def _select_few_shot_examples(self, example: Dict, group_pools: Dict[str, List[Dict]], 
                                 challenging_examples: List[Dict]) -> List[Dict]:
        """Select few-shot examples for the current example.
        
        Args:
            example: Current example
            group_pools: Pools of examples by demographic group
            challenging_examples: List of challenging examples
            
        Returns:
            List of selected few-shot examples
        """
        selected = []
        
        # Get the category of the current example
        category = example.get("category", "")
        
        # Step 1: Include examples from underrepresented groups
        # Find the demographic groups with the fewest examples
        group_sizes = {group: len(pool) for group, pool in group_pools.items()}
        sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1])
        underrepresented_groups = [group for group, _ in sorted_groups[:2]]
        
        # Add one example from each underrepresented group
        for group in underrepresented_groups:
            if group_pools[group]:
                group_examples = [ex for ex in group_pools[group] if ex.get("category") == category]
                if not group_examples:  # If no examples from same category, use any from this group
                    group_examples = group_pools[group]
                
                if group_examples:
                    selected.append(random.choice(group_examples))
        
        # Step 2: Include examples similar to the current example
        # Find examples from the same category
        same_category_examples = []
        for group, pool in group_pools.items():
            same_category_examples.extend([ex for ex in pool if ex.get("category") == category])
        
        # If there are same-category examples, add some
        if same_category_examples:
            # Prioritize examples that are most different from those already selected
            # (simple heuristic: different question_id)
            selected_ids = {ex.get("question_id") for ex in selected}
            different_examples = [ex for ex in same_category_examples if ex.get("question_id") not in selected_ids]
            
            # Add up to 2 examples from the same category
            num_to_add = min(2, self.num_examples - len(selected))
            if different_examples and num_to_add > 0:
                selected.extend(random.sample(different_examples, min(num_to_add, len(different_examples))))
        
        # Step 3: Include challenging examples if needed
        if len(selected) < self.num_examples and challenging_examples:
            # Filter out examples already selected
            selected_ids = {ex.get("example_id") for ex in selected}
            remaining_challenging = [ex for ex in challenging_examples if ex.get("example_id") not in selected_ids]
            
            # Add up to remaining slots with challenging examples
            num_to_add = min(self.num_examples - len(selected), len(remaining_challenging))
            if num_to_add > 0:
                selected.extend(random.sample(remaining_challenging, num_to_add))
        
        # Step 4: If we still need more examples, add random ones from any group
        if len(selected) < self.num_examples:
            all_examples = []
            for pool in group_pools.values():
                all_examples.extend(pool)
            
            # Filter out examples already selected
            selected_ids = {ex.get("example_id") for ex in selected}
            remaining_examples = [ex for ex in all_examples if ex.get("example_id") not in selected_ids]
            
            # Add random examples to fill the remaining slots
            num_to_add = min(self.num_examples - len(selected), len(remaining_examples))
            if num_to_add > 0:
                selected.extend(random.sample(remaining_examples, num_to_add))
        
        # Format the selected examples for the model
        formatted_examples = []
        for ex in selected:
            context = ex.get("context", "")
            question = ex.get("question", "")
            options = [ex.get(f"ans{i}", "") for i in range(3)]
            label = ex.get("label", -1)
            
            # Get the correct answer text
            answer = ex.get(f"ans{label}", "Unknown") if 0 <= label < 3 else "Unknown"
            
            formatted_examples.append({
                "context": context,
                "question": question,
                "options": options,
                "answer": answer,
                "example_id": ex.get("example_id")
            })
        
        return formatted_examples
