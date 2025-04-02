"""
Post-Processing Bias Correction for DeepSeek-R1

This module implements post-processing techniques to correct biases in the model's outputs
without modifying the model itself.

Reference:
Kadhe, S. R., Halimi, A., Rawat, A., & Baracaldo, N. (2023). FairSISA: Ensemble post-processing
to improve fairness of unlearning in LLMs.
"""

import logging
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from model_wrapper import DeepSeekModel

logger = logging.getLogger(__name__)

class BiasCorrector:
    """Implements post-processing bias correction techniques."""
    
    def __init__(self, model: DeepSeekModel, calibration_size: int = 200):
        """Initialize the BiasCorrector.
        
        Args:
            model: DeepSeek model wrapper
            calibration_size: Number of examples to use for calibration
        """
        self.model = model
        self.calibration_size = calibration_size
        self.group_correction_factors = {}
        self.trained_model = None
    
    def enhance(self, data: List[Dict]) -> List[Dict]:
        """Apply post-processing bias correction to enhance fairness.
        
        Args:
            data: List of examples from the BBQ dataset
        
        Returns:
            List of enhanced results with bias-corrected predictions
        """
        logger.info("Applying post-processing bias correction")
        
        # Split data into calibration and evaluation sets
        if len(data) > self.calibration_size:
            random.seed(42)  # For reproducibility
            calibration_data = random.sample(data, self.calibration_size)
            evaluation_data = [ex for ex in data if ex not in calibration_data]
        else:
            # If not enough data, use 80% for calibration and 20% for evaluation
            random.seed(42)
            random.shuffle(data)
            split_idx = int(0.8 * len(data))
            calibration_data = data[:split_idx]
            evaluation_data = data[split_idx:]
        
        logger.info(f"Using {len(calibration_data)} examples for calibration and {len(evaluation_data)} for evaluation")
        
        # Get baseline predictions on calibration data
        calibration_results = self._get_baseline_predictions(calibration_data)
        
        # Train both correction methods on calibration data
        self._train_demographic_parity_correction(calibration_results)
        self._train_logistic_correction(calibration_results)
        
        # Apply bias correction to evaluation data
        corrected_results = []
        for example in tqdm(data, desc="Applying bias correction"):
            # Get baseline prediction first
            context = example.get("context", "")
            question = example.get("question", "")
            options = [example.get(f"ans{i}", "") for i in range(3)]
            
            prediction = self.model.predict(context, question, options)
            
            # Apply post-processing corrections
            corrected_prediction = self._apply_corrections(example, prediction)
            
            # Add predictions to the example
            result = {**example}
            result["model_output"] = {
                "original_prediction": prediction["prediction"],
                "prediction": corrected_prediction["prediction"],
                "raw_output": prediction["raw_output"],
                "prompt_used": prediction["input_text"]
            }
            result["fairness_enhancement"] = {
                "method": "post_processing",
                "correction_method": corrected_prediction["method"]
            }
            corrected_results.append(result)
        
        return corrected_results
    
    def _get_baseline_predictions(self, data: List[Dict]) -> List[Dict]:
        """Get baseline model predictions on the data.
        
        Args:
            data: List of examples
            
        Returns:
            List of results with predictions
        """
        logger.info("Getting baseline predictions for calibration")
        results = []
        
        for example in tqdm(data, desc="Baseline prediction"):
            # Extract data needed for prediction
            context = example.get("context", "")
            question = example.get("question", "")
            options = [example.get(f"ans{i}", "") for i in range(3)]
            
            # Make prediction using default prompt
            prediction = self.model.predict(context, question, options)
            
            # Add prediction to the example
            result = {**example}
            result["model_output"] = {
                "prediction": prediction["prediction"],
                "raw_output": prediction["raw_output"],
                "prompt_used": prediction["input_text"]
            }
            results.append(result)
        
        return results
    
    def _train_demographic_parity_correction(self, calibration_results: List[Dict]) -> None:
        """Train demographic parity correction factors.
        
        This method computes correction factors to balance prediction rates across demographic groups.
        
        Args:
            calibration_results: List of calibration results with predictions
        """
        logger.info("Training demographic parity correction")
        
        # Extract demographic groups and predictions
        group_predictions = defaultdict(list)
        
        for result in calibration_results:
            # Extract demographic information from answer_info
            answer_info = result.get("answer_info", {})
            
            # Get the predicted answer
            pred = result.get("model_output", {}).get("prediction", "")
            
            # Map prediction to group
            pred_group = None
            for i in range(3):
                ans_key = f"ans{i}"
                ans_text = result.get(ans_key, "")
                if pred in ans_text and ans_key in answer_info:
                    pred_group = answer_info[ans_key][1] if len(answer_info[ans_key]) > 1 else "unknown"
                    break
            
            if pred_group:
                group_predictions[pred_group].append(pred)
        
        # Compute group prediction rates
        group_rates = {}
        total_predictions = sum(len(preds) for preds in group_predictions.values())
        
        for group, preds in group_predictions.items():
            group_rates[group] = len(preds) / total_predictions if total_predictions > 0 else 0
        
        # Compute correction factors to achieve demographic parity
        # Ideally, each group should have equal representation in predictions
        ideal_rate = 1.0 / len(group_predictions) if group_predictions else 0
        
        self.group_correction_factors = {}
        for group, rate in group_rates.items():
            if rate > 0:
                # If a group is over-represented, reduce its probability
                # If under-represented, increase its probability
                self.group_correction_factors[group] = ideal_rate / rate
            else:
                self.group_correction_factors[group] = 1.0
        
        logger.info(f"Computed correction factors: {self.group_correction_factors}")
    
    def _train_logistic_correction(self, calibration_results: List[Dict]) -> None:
        """Train a logistic regression model to correct biased predictions.
        
        This method trains a model that predicts the correct answer given the raw model prediction
        and demographic features.
        
        Args:
            calibration_results: List of calibration results with predictions
        """
        logger.info("Training logistic regression correction model")
        
        # Extract features and labels
        features = []
        labels = []
        
        for result in calibration_results:
            # Extract categorical features
            category = result.get("category", "unknown")
            context_condition = result.get("context_condition", "unknown")
            question_polarity = result.get("question_polarity", "unknown")
            
            # Get the predicted answer and true label
            pred_text = result.get("model_output", {}).get("prediction", "")
            pred_idx = -1
            
            for i in range(3):
                ans_key = f"ans{i}"
                ans_text = result.get(ans_key, "")
                if pred_text in ans_text:
                    pred_idx = i
                    break
            
            # Get true label
            true_label = result.get("label", -1)
            
            # Skip examples with invalid predictions or labels
            if pred_idx == -1 or true_label == -1:
                continue
            
            # Create feature vector
            feature = {
                "category": category,
                "context_condition": context_condition,
                "question_polarity": question_polarity,
                "predicted_idx": pred_idx
            }
            
            features.append(feature)
            labels.append(true_label)
        
        # Convert to DataFrame for easier processing
        if not features:
            logger.warning("No valid features for logistic correction. Skipping.")
            return
        
        df = pd.DataFrame(features)
        
        # One-hot encode categorical features
        categorical_cols = ["category", "context_condition", "question_polarity"]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(df[categorical_cols])
        
        # Combine with numeric features
        X = np.hstack([encoded_cats, df[["predicted_idx"]].values])
        y = np.array(labels)
        
        # Train logistic regression model
        try:
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X, y)
            self.trained_model = {
                "model": model,
                "encoder": encoder,
                "categorical_cols": categorical_cols
            }
            logger.info("Logistic correction model trained successfully")
        except Exception as e:
            logger.error(f"Error training logistic correction model: {e}")
            self.trained_model = None
    
    def _apply_corrections(self, example: Dict, prediction: Dict) -> Dict:
        """Apply bias corrections to the prediction.
        
        Args:
            example: Example data
            prediction: Original model prediction
            
        Returns:
            Corrected prediction
        """
        # Try both correction methods and select the best one
        demographic_correction = self._apply_demographic_parity_correction(example, prediction)
        logistic_correction = self._apply_logistic_correction(example, prediction)
        
        # Choose which correction to apply
        # If the example is ambiguous, prefer demographic parity to avoid biased predictions
        if example.get("context_condition") == "ambig":
            return demographic_correction
        else:
            # For disambiguated examples, prefer logistic correction for better accuracy
            return logistic_correction if logistic_correction["confidence"] > 0.6 else demographic_correction
    
    def _apply_demographic_parity_correction(self, example: Dict, prediction: Dict) -> Dict:
        """Apply demographic parity correction.
        
        Args:
            example: Example data
            prediction: Original model prediction
            
        Returns:
            Corrected prediction
        """
        pred_text = prediction["prediction"]
        
        # Get all answer options and their groups
        options = []
        groups = []
        for i in range(3):
            ans_key = f"ans{i}"
            ans_text = example.get(ans_key, "")
            options.append(ans_text)
            
            # Get group from answer_info
            ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
            group = ans_info[1] if len(ans_info) > 1 else "unknown"
            groups.append(group)
        
        # Find which option matches the prediction
        pred_idx = -1
        for i, opt in enumerate(options):
            if pred_text in opt:
                pred_idx = i
                break
        
        # If original prediction is valid, apply correction
        if pred_idx != -1:
            pred_group = groups[pred_idx]
            
            # Get correction factor for this group
            correction_factor = self.group_correction_factors.get(pred_group, 1.0)
            
            # If correction strongly suggests reducing this group's probability
            if correction_factor < 0.7:
                # Find an alternative option with higher correction factor
                alternative_scores = []
                for i, group in enumerate(groups):
                    if i != pred_idx:
                        alt_factor = self.group_correction_factors.get(group, 1.0)
                        alternative_scores.append((i, alt_factor))
                
                # Select the option with the highest correction factor
                if alternative_scores:
                    best_alt = max(alternative_scores, key=lambda x: x[1])
                    if best_alt[1] > correction_factor:
                        pred_idx = best_alt[0]
        
        # Get the final corrected prediction
        corrected_pred = options[pred_idx] if 0 <= pred_idx < len(options) else pred_text
        
        return {
            "prediction": corrected_pred,
            "method": "demographic_parity",
            "confidence": 0.8  # Fixed confidence for demographic parity
        }
    
    def _apply_logistic_correction(self, example: Dict, prediction: Dict) -> Dict:
        """Apply logistic regression correction.
        
        Args:
            example: Example data
            prediction: Original model prediction
            
        Returns:
            Corrected prediction
        """
        # If no trained model is available, return original prediction
        if not self.trained_model:
            return {
                "prediction": prediction["prediction"],
                "method": "original",
                "confidence": 0.5
            }
        
        # Extract features
        category = example.get("category", "unknown")
        context_condition = example.get("context_condition", "unknown")
        question_polarity = example.get("question_polarity", "unknown")
        
        # Get predicted index
        pred_text = prediction["prediction"]
        pred_idx = -1
        for i in range(3):
            ans_key = f"ans{i}"
            ans_text = example.get(ans_key, "")
            if pred_text in ans_text:
                pred_idx = i
                break
        
        # If can't map prediction to an index, return original prediction
        if pred_idx == -1:
            return {
                "prediction": pred_text,
                "method": "original",
                "confidence": 0.5
            }
        
        # Create feature vector
        feature_df = pd.DataFrame([{
            "category": category,
            "context_condition": context_condition,
            "question_polarity": question_polarity,
            "predicted_idx": pred_idx
        }])
        
        # One-hot encode categorical features
        model = self.trained_model["model"]
        encoder = self.trained_model["encoder"]
        categorical_cols = self.trained_model["categorical_cols"]
        
        encoded_cats = encoder.transform(feature_df[categorical_cols])
        X = np.hstack([encoded_cats, feature_df[["predicted_idx"]].values])
        
        # Predict correct label
        try:
            corrected_idx = model.predict(X)[0]
            # Get confidence from probabilities
            probs = model.predict_proba(X)[0]
            confidence = probs[corrected_idx]
            
            # Get the corrected prediction text
            corrected_pred = example.get(f"ans{corrected_idx}", pred_text)
            
            return {
                "prediction": corrected_pred,
                "method": "logistic_regression",
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Error applying logistic correction: {e}")
            return {
                "prediction": pred_text,
                "method": "original",
                "confidence": 0.5
            }
