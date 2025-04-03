"""
Fairness Ensemble for DeepSeek-R1

This module implements an ensemble approach that combines multiple fairness enhancement methods
to achieve comprehensive bias mitigation.
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter

from model_wrapper import DeepSeekModel
from prompt_optimization import FairPromptOptimizer
from few_shot_selection import FewShotSelector
from post_processing import BiasCorrector

logger = logging.getLogger(__name__)

class FairnessEnsemble:
    """Implements ensemble approach combining multiple fairness enhancement methods."""
    
    def __init__(self, model: DeepSeekModel, 
                 use_prompt_optimization: bool = True,
                 use_few_shot: bool = True, 
                 use_post_processing: bool = True,
                 meta_llm: Optional[str] = None,
                 voting_strategy: str = "weighted"):
        """Initialize the FairnessEnsemble.
        
        Args:
            model: DeepSeek model wrapper
            use_prompt_optimization: Whether to use prompt optimization
            use_few_shot: Whether to use few-shot selection
            use_post_processing: Whether to use post-processing
            meta_llm: Optional meta-LLM to use for prompt optimization
            voting_strategy: Strategy for combining predictions ("majority", "weighted", or "contextual")
        """
        self.model = model
        self.use_prompt_optimization = use_prompt_optimization
        self.use_few_shot = use_few_shot
        self.use_post_processing = use_post_processing
        self.voting_strategy = voting_strategy
        
        # Initialize component enhancers
        if self.use_prompt_optimization:
            self.prompt_optimizer = FairPromptOptimizer(model, meta_llm=meta_llm)
        
        if self.use_few_shot:
            self.few_shot_selector = FewShotSelector(model)
        
        if self.use_post_processing:
            self.bias_corrector = BiasCorrector(model)
        
        # Method weights for weighted voting
        self.method_weights = {
            "prompt_optimization": 0.3,
            "few_shot_selection": 0.3,
            "post_processing": 0.4
        }
    
    def enhance(self, data: List[Dict]) -> List[Dict]:
        """Apply ensemble fairness enhancement.
        
        Args:
            data: List of examples from the BBQ dataset
        
        Returns:
            List of enhanced results with ensemble predictions
        """
        logger.info("Applying ensemble fairness enhancement")
        
        # Split data if too large
        if len(data) > 300:
            sample_size = 300
            logger.info(f"Data is large ({len(data)} examples), using a sample of {sample_size} for calibration")
            random.seed(42)
            calibration_data = random.sample(data, sample_size)
        else:
            calibration_data = data
        
        # Apply individual enhancement methods to calibration data
        enhancement_results = {}
        
        if self.use_prompt_optimization:
            logger.info("Running prompt optimization...")
            prompt_results = self.prompt_optimizer.enhance(calibration_data)
            enhancement_results["prompt_optimization"] = prompt_results
        
        if self.use_few_shot:
            logger.info("Running few-shot selection...")
            few_shot_results = self.few_shot_selector.enhance(calibration_data)
            enhancement_results["few_shot_selection"] = few_shot_results
        
        if self.use_post_processing:
            logger.info("Running post-processing...")
            post_processing_results = self.bias_corrector.enhance(calibration_data)
            enhancement_results["post_processing"] = post_processing_results
        
        # Evaluate method performance on calibration data
        if self.voting_strategy == "weighted":
            self._update_method_weights(enhancement_results, calibration_data)
        
        # Apply ensemble to all data
        logger.info("Applying ensemble to all examples...")
        ensemble_results = []
        
        # Process each example with all enabled methods
        for example in tqdm(data, desc="Ensemble processing"):
            # Get predictions from each method
            predictions = self._get_all_predictions(example)
            
            # Combine predictions using the selected voting strategy
            ensemble_prediction = self._combine_predictions(predictions, example)
            
            # Format the result
            result = {**example}
            result["model_output"] = {
                "prediction": ensemble_prediction["final_prediction"],
                "individual_predictions": predictions,
                "voting_strategy": self.voting_strategy,
                "weights": ensemble_prediction.get("weights", {})
            }
            result["fairness_enhancement"] = {
                "method": "ensemble",
                "component_methods": list(predictions.keys())
            }
            ensemble_results.append(result)
        
        return ensemble_results
    
    def _get_all_predictions(self, example: Dict) -> Dict[str, Any]:
        """Get predictions from all enabled enhancement methods.
        
        Args:
            example: Example data
            
        Returns:
            Dictionary mapping method names to predictions
        """
        predictions = {}
        
        # Extract common data
        context = example.get("context", "")
        question = example.get("question", "")
        options = [example.get(f"ans{i}", "") for i in range(3)]
        
        # Get baseline prediction
        baseline = self.model.predict(context, question, options)
        predictions["baseline"] = baseline["prediction"]
        
        # Get predictions from each method
        if self.use_prompt_optimization:
            # Use the optimized prompt from calibration
            optimized_prompt = self.prompt_optimizer.prompt_templates[0]  # Use first template as fallback
            try:
                prompt_pred = self.model.predict(context, question, options, prompt_template=optimized_prompt)
                predictions["prompt_optimization"] = prompt_pred["prediction"]
            except Exception as e:
                logger.error(f"Error in prompt optimization prediction: {e}")
                predictions["prompt_optimization"] = baseline["prediction"]
        
        if self.use_few_shot:
            # Select few-shot examples
            try:
                few_shot_examples = self.few_shot_selector._select_few_shot_examples(
                    example, 
                    self.few_shot_selector._create_group_pools(
                        [example], 
                        self.few_shot_selector._identify_groups([example])
                    ),
                    []
                )
                few_shot_pred = self.model.predict(
                    context, 
                    question, 
                    options, 
                    prompt_template=self.few_shot_selector.base_prompt,
                    few_shot_examples=few_shot_examples
                )
                predictions["few_shot_selection"] = few_shot_pred["prediction"]
            except Exception as e:
                logger.error(f"Error in few-shot prediction: {e}")
                predictions["few_shot_selection"] = baseline["prediction"]
        
        if self.use_post_processing:
            # Apply post-processing
            try:
                corrected = self.bias_corrector._apply_corrections(example, baseline)
                predictions["post_processing"] = corrected["prediction"]
            except Exception as e:
                logger.error(f"Error in post-processing: {e}")
                predictions["post_processing"] = baseline["prediction"]
        
        return predictions
    
    def _combine_predictions(self, predictions: Dict[str, str], example: Dict) -> Dict[str, Any]:
        """Combine predictions using the selected voting strategy.
        
        Args:
            predictions: Dictionary mapping method names to predictions
            example: Original example data
            
        Returns:
            Dictionary with final prediction and metadata
        """
        result = {
            "final_prediction": "",
            "voting_method": self.voting_strategy
        }
        
        # Remove baseline from voting if we have other methods
        voting_predictions = {k: v for k, v in predictions.items() if k != "baseline"}
        if not voting_predictions:
            result["final_prediction"] = predictions.get("baseline", "")
            return result
        
        # Map predictions to answer indices for consistent comparison
        prediction_indices = {}
        for method, pred in voting_predictions.items():
            idx = -1
            for i in range(3):
                ans_text = example.get(f"ans{i}", "")
                if pred in ans_text:
                    idx = i
                    break
            prediction_indices[method] = idx
        
        # Apply the selected voting strategy
        if self.voting_strategy == "majority":
            # Simple majority vote
            votes = Counter(idx for method, idx in prediction_indices.items() if idx != -1)
            if votes:
                majority_idx = votes.most_common(1)[0][0]
                result["final_prediction"] = example.get(f"ans{majority_idx}", predictions.get("baseline", ""))
            else:
                result["final_prediction"] = predictions.get("baseline", "")
        
        elif self.voting_strategy == "weighted":
            # Weighted vote
            weights = {method: self.method_weights.get(method, 0.0) for method in voting_predictions}
            result["weights"] = weights
            
            # Count weighted votes for each answer option
            weighted_votes = defaultdict(float)
            for method, idx in prediction_indices.items():
                if idx != -1:
                    weighted_votes[idx] += weights[method]
            
            # Select answer with highest weighted vote
            if weighted_votes:
                max_idx = max(weighted_votes.items(), key=lambda x: x[1])[0]
                result["final_prediction"] = example.get(f"ans{max_idx}", predictions.get("baseline", ""))
            else:
                result["final_prediction"] = predictions.get("baseline", "")
        
        elif self.voting_strategy == "contextual":
            # Contextual voting: choose method based on example properties
            context_condition = example.get("context_condition", "")
            
            if context_condition == "ambig":
                # For ambiguous contexts, prefer prompt optimization
                if "prompt_optimization" in prediction_indices and prediction_indices["prompt_optimization"] != -1:
                    selected_method = "prompt_optimization"
                elif "post_processing" in prediction_indices and prediction_indices["post_processing"] != -1:
                    selected_method = "post_processing"
                else:
                    selected_method = next(iter(voting_predictions.keys()))
            else:
                # For disambiguated contexts, prefer few-shot or post-processing
                if "few_shot_selection" in prediction_indices and prediction_indices["few_shot_selection"] != -1:
                    selected_method = "few_shot_selection"
                elif "post_processing" in prediction_indices and prediction_indices["post_processing"] != -1:
                    selected_method = "post_processing"
                else:
                    selected_method = next(iter(voting_predictions.keys()))
            
            result["selected_method"] = selected_method
            idx = prediction_indices.get(selected_method, -1)
            
            if idx != -1:
                result["final_prediction"] = example.get(f"ans{idx}", predictions.get(selected_method, ""))
            else:
                result["final_prediction"] = predictions.get(selected_method, predictions.get("baseline", ""))
        
        else:
            # Default to first available prediction
            result["final_prediction"] = next(iter(voting_predictions.values()), predictions.get("baseline", ""))
        
        return result
    
    def _update_method_weights(self, enhancement_results: Dict[str, List[Dict]], calibration_data: List[Dict]) -> None:
        """Update method weights based on performance on calibration data.
        
        Args:
            enhancement_results: Dictionary mapping method names to enhanced results
            calibration_data: Original calibration data
        """
        logger.info("Updating method weights based on calibration performance")
        
        # Calculate accuracy for each method
        method_accuracy = {}
        
        # Create lookup for true labels
        example_id_to_label = {ex.get("example_id"): ex.get("label") for ex in calibration_data}
        
        for method, results in enhancement_results.items():
            correct = 0
            total = 0
            
            for result in results:
                example_id = result.get("example_id")
                true_label = example_id_to_label.get(example_id, -1)
                
                if true_label == -1:
                    continue
                
                # Get prediction and convert to index
                pred = result.get("model_output", {}).get("prediction", "")
                pred_idx = -1
                
                for i in range(3):
                    ans_text = result.get(f"ans{i}", "")
                    if pred in ans_text:
                        pred_idx = i
                        break
                
                if pred_idx == true_label:
                    correct += 1
                total += 1
            
            if total > 0:
                method_accuracy[method] = correct / total
            else:
                method_accuracy[method] = 0.0
        
        logger.info(f"Method accuracies: {method_accuracy}")
        
        # Calculate fairness metrics for each method
        method_fairness = {}
        
        for method, results in enhancement_results.items():
            # Extract demographic groups and predictions
            groups = []
            preds = []
            labels = []
            
            for result in results:
                # Skip if we can't determine the prediction
                pred = result.get("model_output", {}).get("prediction", "")
                pred_idx = -1
                
                for i in range(3):
                    ans_text = result.get(f"ans{i}", "")
                    if pred in ans_text:
                        pred_idx = i
                        break
                
                if pred_idx == -1:
                    continue
                
                # Extract demographic information from answer_info
                ans_key = f"ans{pred_idx}"
                answer_info = result.get("answer_info", {})
                
                if ans_key in answer_info:
                    group = answer_info[ans_key][1] if len(answer_info[ans_key]) > 1 else "unknown"
                    groups.append(group)
                    preds.append(pred_idx)
                    
                    # Get true label
                    true_label = result.get("label", -1)
                    labels.append(true_label)
            
            # Compute fairness gap
            group_to_examples = defaultdict(list)
            group_to_correct = defaultdict(int)
            
            for i, group in enumerate(groups):
                if labels[i] == -1:
                    continue
                
                group_to_examples[group].append(i)
                if preds[i] == labels[i]:
                    group_to_correct[group] += 1
            
            # Group accuracy
            group_accuracy = {}
            for group, examples in group_to_examples.items():
                if examples:
                    group_accuracy[group] = group_to_correct[group] / len(examples)
                else:
                    group_accuracy[group] = 0.0
            
            # Compute fairness gap (difference between max and min group accuracy)
            if group_accuracy:
                max_acc = max(group_accuracy.values())
                min_acc = min(group_accuracy.values())
                fairness_gap = max_acc - min_acc
                
                # Lower gap means better fairness
                method_fairness[method] = 1.0 - fairness_gap
            else:
                method_fairness[method] = 0.0
        
        logger.info(f"Method fairness: {method_fairness}")
        
        # Combine accuracy and fairness to update weights
        # Weight = 0.5 * accuracy + 0.5 * fairness
        for method in self.method_weights:
            if method in method_accuracy and method in method_fairness:
                self.method_weights[method] = 0.5 * method_accuracy[method] + 0.5 * method_fairness[method]
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.method_weights.values())
        if weight_sum > 0:
            self.method_weights = {method: weight / weight_sum for method, weight in self.method_weights.items()}
        
        logger.info(f"Updated method weights: {self.method_weights}")
