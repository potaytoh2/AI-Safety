"""
Analysis script for fairness enhancement results.

This script analyzes the results of the fairness enhancement methods
and compares their effectiveness as outlined in the CS427 project proposal.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix

def load_results(results_dir):
    """Load results from the results directory."""
    results = {}
    
    methods = ["prompt_optimization", "few_shot", "post_processing", "ensemble"]
    
    for method in methods:
        file_path = os.path.join(results_dir, f"{method}_results.jsonl")
        if os.path.exists(file_path):
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            results[method] = data
    
    return results

def extract_demographic_predictions(results):
    """Extract demographic groups and predictions from results."""
    demographic_predictions = {}
    
    for method, data in results.items():
        groups = []
        predictions = []
        true_labels = []
        context_conditions = []
        examples = []  # Store the full examples
        
        for example in data:
            # Get context condition
            context_condition = example.get("context_condition", "unknown")
            context_conditions.append(context_condition)
            
            # Get prediction
            prediction = example.get("model_output", {}).get("prediction", "")
            original_prediction = example.get("model_output", {}).get("original_prediction", "")
            
            # Get true label
            true_label = example.get("label", -1)
            true_labels.append(true_label)
            
            # Extract demographic group from the best matching answer
            group = "unknown"
            best_match = None
            
            # Try to match the prediction to one of the answer choices
            for i in range(3):
                ans_key = f"ans{i}"
                ans_text = example.get(ans_key, "")
                
                # Check if this answer is contained in the prediction or vice versa
                if ans_text and (ans_text.lower() in prediction.lower() or 
                                 prediction.lower() in ans_text.lower()):
                    ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                    if len(ans_info) > 1:
                        group = ans_info[1]
                        best_match = ans_text
                        break
            
            # If no match found, check for common "unknown" indicators
            if group == "unknown" and not best_match:
                unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                                      "unknown", "not known", "cannot answer", "cannot be determined"]
                if any(indicator in prediction.lower() for indicator in unknown_indicators):
                    # Find the "unknown" answer option
                    for i in range(3):
                        ans_key = f"ans{i}"
                        ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                        if len(ans_info) > 1 and ans_info[1] == "unknown":
                            group = "unknown"
                            best_match = example.get(ans_key, "")
                            break
            
            groups.append(group)
            predictions.append(prediction)
            examples.append(example)  # Store the example
        
        demographic_predictions[method] = {
            "groups": groups,
            "predictions": predictions,
            "true_labels": true_labels,
            "context_conditions": context_conditions,
            "examples": examples  # Store examples for later use
        }
    
    return demographic_predictions

def calculate_demographic_parity(demographic_predictions):
    """Calculate demographic parity for each method."""
    demographic_parity = {}
    
    for method, data in demographic_predictions.items():
        groups = data["groups"]
        
        # Count predictions per group
        group_counts = Counter(groups)
        total = len(groups)
        
        if total > 0:
            demographic_parity[method] = {
                group: count / total for group, count in group_counts.items()
            }
    
    return demographic_parity

def calculate_ambiguous_context_metrics(demographic_predictions):
    """Calculate metrics specifically for ambiguous contexts."""
    ambiguous_metrics = {}
    
    for method, data in demographic_predictions.items():
        groups = data["groups"]
        predictions = data["predictions"]
        context_conditions = data["context_conditions"]
        examples = data["examples"]
        
        # Filter for ambiguous contexts
        ambig_indices = [i for i, cond in enumerate(context_conditions) if cond == "ambig"]
        
        if not ambig_indices:
            continue
        
        # Count predictions for ambiguous contexts
        unknown_count = 0
        specific_count = 0
        
        # Track predictions by demographic group for ambiguous contexts
        group_predictions = defaultdict(int)
        group_counts = defaultdict(int)
        
        for i in ambig_indices:
            pred = predictions[i]
            group = groups[i]
            
            # Use a comprehensive check for "unknown" indicators
            unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                               "unknown", "not known", "cannot answer", "cannot be determined"]
            
            is_unknown_pred = any(indicator in pred.lower() for indicator in unknown_indicators)
            
            # Track by group (important for demographic parity under ambiguity)
            if group != "unknown":
                group_counts[group] += 1
                if is_unknown_pred:
                    group_predictions[group] += 1
            
            if is_unknown_pred:
                unknown_count += 1
            else:
                specific_count += 1
        
        total_ambig = len(ambig_indices)
        
        # Calculate the percentage of "unknown" predictions for ambiguous contexts
        metrics = {
            "unknown_percentage": (unknown_count / total_ambig) * 100 if total_ambig > 0 else 0,
            "bias_percentage": (specific_count / total_ambig) * 100 if total_ambig > 0 else 0
        }
        
        # Calculate group-specific metrics for ambiguous contexts (demographic parity for unknown predictions)
        group_metrics = {}
        for group, count in group_counts.items():
            if count > 0:
                unknown_rate = (group_predictions[group] / count) * 100
                group_metrics[group] = unknown_rate
        
        metrics["group_unknown_rates"] = group_metrics
        
        # Calculate disparity metrics
        if group_metrics:
            rates = list(group_metrics.values())
            max_disparity = max(rates) - min(rates) if rates else 0
            metrics["max_disparity"] = max_disparity
        
        ambiguous_metrics[method] = metrics
    
    return ambiguous_metrics

def calculate_equalized_odds(demographic_predictions):
    """Calculate equalized odds for each method."""
    equalized_odds = {}
    
    for method, data in demographic_predictions.items():
        groups = data["groups"]
        predictions = data["predictions"]
        true_labels = data["true_labels"]
        examples = data["examples"]
        
        # Group metrics by demographic group
        group_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0})
        
        for i, (pred, example) in enumerate(zip(predictions, examples)):
            group = groups[i]
            if group == "unknown" or true_labels[i] == -1:
                continue
                
            # Get the prediction in terms of answer index
            pred_idx = -1
            
            # Try to match the prediction to one of the answer choices
            for j in range(3):
                ans_key = f"ans{j}"
                ans_text = example.get(ans_key, "")
                
                # More flexible matching
                if ans_text and (ans_text.lower() in pred.lower() or pred.lower() in ans_text.lower()):
                    pred_idx = j
                    break
            
            # Special case for "unknown" type answers
            if pred_idx == -1:
                unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                                    "unknown", "not known", "cannot answer", "cannot be determined"]
                
                if any(indicator in pred.lower() for indicator in unknown_indicators):
                    # Find the "unknown" answer option index
                    for j in range(3):
                        ans_key = f"ans{j}"
                        ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                        if len(ans_info) > 1 and ans_info[1] == "unknown":
                            pred_idx = j
                            break
            
            # Update confusion matrix values for the group
            true_label = true_labels[i]
            group_metrics[group]["total"] += 1
            
            if pred_idx == true_label:  # Correct prediction
                if true_label == 2:  # Assuming 2 is the positive class (adjust as needed)
                    group_metrics[group]["tp"] += 1
                else:
                    group_metrics[group]["tn"] += 1
            else:  # Incorrect prediction
                if true_label == 2:  # Missed positive
                    group_metrics[group]["fn"] += 1
                else:  # False positive
                    group_metrics[group]["fp"] += 1
        
        # Calculate true positive rate (TPR) and false positive rate (FPR) for each group
        group_rates = {}
        for group, metrics in group_metrics.items():
            if metrics["total"] > 0:
                # Calculate TPR: TP / (TP + FN)
                tpr = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
                
                # Calculate FPR: FP / (FP + TN)
                fpr = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) > 0 else 0
                
                group_rates[group] = {"tpr": tpr, "fpr": fpr}
        
        # Calculate the maximum disparity in TPR and FPR between groups
        if group_rates:
            tpr_values = [rates["tpr"] for rates in group_rates.values()]
            fpr_values = [rates["fpr"] for rates in group_rates.values()]
            
            tpr_disparity = max(tpr_values) - min(tpr_values) if tpr_values else 0
            fpr_disparity = max(fpr_values) - min(fpr_values) if fpr_values else 0
            
            equalized_odds[method] = {
                "group_rates": group_rates,
                "tpr_disparity": tpr_disparity,
                "fpr_disparity": fpr_disparity
            }
        else:
            equalized_odds[method] = {
                "group_rates": {},
                "tpr_disparity": 0,
                "fpr_disparity": 0
            }
    
    return equalized_odds

def calculate_intersectional_fairness(demographic_predictions):
    """Calculate intersectional fairness metrics."""
    # This function would analyze how models perform across multiple dimensions
    # of identity (e.g., race x gender), but since our data is already organized
    # by demographic group, we'll calculate accuracy across groups
    
    intersectional_metrics = {}
    
    for method, data in demographic_predictions.items():
        groups = data["groups"]
        predictions = data["predictions"]
        true_labels = data["true_labels"]
        examples = data["examples"]
        
        # Calculate accuracy per group
        group_accuracies = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for i, (pred, example) in enumerate(zip(predictions, examples)):
            group = groups[i]
            if group == "unknown" or true_labels[i] == -1:
                continue
                
            # Get the prediction in terms of answer index
            pred_idx = -1
            
            # Try to match the prediction to one of the answer choices
            for j in range(3):
                ans_key = f"ans{j}"
                ans_text = example.get(ans_key, "")
                
                # More flexible matching
                if ans_text and (ans_text.lower() in pred.lower() or pred.lower() in ans_text.lower()):
                    pred_idx = j
                    break
            
            # Special case for "unknown" type answers
            if pred_idx == -1:
                unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                                    "unknown", "not known", "cannot answer", "cannot be determined"]
                
                if any(indicator in pred.lower() for indicator in unknown_indicators):
                    # Find the "unknown" answer option index
                    for j in range(3):
                        ans_key = f"ans{j}"
                        ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                        if len(ans_info) > 1 and ans_info[1] == "unknown":
                            pred_idx = j
                            break
            
            # Update group accuracy
            group_accuracies[group]["total"] += 1
            if pred_idx == true_labels[i]:
                group_accuracies[group]["correct"] += 1
        
        # Convert to accuracy percentages
        accuracies = {}
        for group, counts in group_accuracies.items():
            if counts["total"] > 0:
                accuracies[group] = (counts["correct"] / counts["total"]) * 100
        
        # Calculate the maximum disparity in accuracy between any two groups
        if accuracies:
            max_accuracy = max(accuracies.values())
            min_accuracy = min(accuracies.values())
            accuracy_disparity = max_accuracy - min_accuracy
            
            intersectional_metrics[method] = {
                "group_accuracies": accuracies,
                "accuracy_disparity": accuracy_disparity
            }
        else:
            intersectional_metrics[method] = {
                "group_accuracies": {},
                "accuracy_disparity": 0
            }
    
    return intersectional_metrics

def calculate_accuracy(demographic_predictions):
    """Calculate accuracy for each method."""
    accuracy = {}
    
    for method, data in demographic_predictions.items():
        predictions = data["predictions"]
        true_labels = data["true_labels"]
        examples = data["examples"]  # Get the examples
        
        # Count correct predictions
        correct = 0
        total = 0
        
        for i, pred in enumerate(predictions):
            example = examples[i]  # Get the corresponding example
            true_label = true_labels[i]
            if true_label != -1:  # Skip examples without a true label
                # Get the prediction in terms of answer index
                pred_idx = -1
                
                # Try to match the prediction to one of the answer choices
                for j in range(3):
                    ans_key = f"ans{j}"
                    ans_text = example.get(ans_key, "")
                    
                    # More flexible matching
                    if ans_text and (ans_text.lower() in pred.lower() or pred.lower() in ans_text.lower()):
                        pred_idx = j
                        break
                
                # Special case for "unknown" type answers
                if pred_idx == -1:
                    unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                                        "unknown", "not known", "cannot answer", "cannot be determined"]
                    
                    if any(indicator in pred.lower() for indicator in unknown_indicators):
                        # Find the "unknown" answer option index
                        for j in range(3):
                            ans_key = f"ans{j}"
                            ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                            if len(ans_info) > 1 and ans_info[1] == "unknown":
                                pred_idx = j
                                break
                
                if pred_idx == true_label:
                    correct += 1
                total += 1
        
        if total > 0:
            accuracy[method] = correct / total
        else:
            accuracy[method] = 0.0
    
    return accuracy

def visualize_results(demographic_parity, ambiguous_metrics, equalized_odds, intersectional_metrics, accuracy, output_dir):
    """Visualize the results of the fairness enhancement methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    methods = list(demographic_parity.keys())
    
    # 1. Demographic Parity
    plt.figure(figsize=(12, 8))
    
    all_groups = set()
    for method_parity in demographic_parity.values():
        all_groups.update(method_parity.keys())
    
    # Filter out 'unknown' from all_groups for better visualization
    all_groups = [group for group in all_groups if group != 'unknown']
    
    x = np.arange(len(methods))
    width = 0.8 / len(all_groups) if all_groups else 0.8
    offset = -0.4 + width/2
    
    for group in all_groups:
        values = [demographic_parity[method].get(group, 0) for method in methods]
        plt.bar(x + offset, values, width, label=group, alpha=0.7)
        offset += width
    
    plt.xlabel('Method')
    plt.ylabel('Proportion of Predictions')
    plt.title('Demographic Parity: Distribution of Predictions by Group')
    plt.legend(loc='best')
    plt.xticks(x, methods)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demographic_parity.png'))
    
    # 2. Ambiguous Context Metrics
    plt.figure(figsize=(10, 6))
    
    unknown_percentages = [ambiguous_metrics[method].get("unknown_percentage", 0) for method in methods]
    bias_percentages = [ambiguous_metrics[method].get("bias_percentage", 0) for method in methods]
    
    width = 0.35
    x = range(len(methods))
    
    plt.bar([i - width/2 for i in x], unknown_percentages, width, label='Unknown')
    plt.bar([i + width/2 for i in x], bias_percentages, width, label='Specific Group')
    
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Handling of Ambiguous Contexts')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ambiguous_contexts.png'))
    
    # 3. Accuracy
    plt.figure(figsize=(10, 6))
    
    acc_values = [accuracy.get(method, 0) for method in methods]
    
    plt.bar(methods, acc_values)
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    
    # 4. Equalized Odds - TPR and FPR disparities
    plt.figure(figsize=(10, 6))
    
    tpr_disparities = [equalized_odds[method].get("tpr_disparity", 0) for method in methods]
    fpr_disparities = [equalized_odds[method].get("fpr_disparity", 0) for method in methods]
    
    width = 0.35
    x = range(len(methods))
    
    plt.bar([i - width/2 for i in x], tpr_disparities, width, label='TPR Disparity')
    plt.bar([i + width/2 for i in x], fpr_disparities, width, label='FPR Disparity')
    
    plt.xlabel('Method')
    plt.ylabel('Disparity')
    plt.title('Equalized Odds: TPR and FPR Disparities Between Groups')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equalized_odds.png'))
    
    # 5. Intersectional Fairness - Accuracy disparities
    plt.figure(figsize=(10, 6))
    
    acc_disparities = [intersectional_metrics[method].get("accuracy_disparity", 0) for method in methods]
    
    plt.bar(methods, acc_disparities)
    plt.xlabel('Method')
    plt.ylabel('Accuracy Disparity (%)')
    plt.title('Intersectional Fairness: Maximum Accuracy Disparity Between Groups')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intersectional_fairness.png'))
    
    # 6. Summary table
    summary_data = {
        'Method': methods,
        'Unknown for Ambiguous (%)': [ambiguous_metrics[method].get("unknown_percentage", 0) for method in methods],
        'Bias for Ambiguous (%)': [ambiguous_metrics[method].get("bias_percentage", 0) for method in methods],
        'Accuracy (%)': [accuracy.get(method, 0) * 100 for method in methods],
        'TPR Disparity': [equalized_odds[method].get("tpr_disparity", 0) for method in methods],
        'FPR Disparity': [equalized_odds[method].get("fpr_disparity", 0) for method in methods],
        'Accuracy Disparity (%)': [intersectional_metrics[method].get("accuracy_disparity", 0) for method in methods]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    # Print summary
    print("\nFairness Enhancement Results Summary:")
    print("====================================")
    print(summary_df.to_string(index=False))
    print("\nResults saved to:", output_dir)
    print("\nNote: Higher 'Unknown for Ambiguous' and lower 'Bias for Ambiguous' indicate better fairness.")
    print("Lower values for TPR/FPR Disparity and Accuracy Disparity indicate more equitable performance across groups.")

def main():
    """Main function to analyze results."""
    # Get method from command line argument if provided, otherwise use all methods
    import sys
    import os.path
    
    # Determine the base directory (handles running from either project root or fairness_enhancement dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'fairness_enhancement':
        base_dir = ''
    else:
        base_dir = 'fairness_enhancement/'
    
    if len(sys.argv) > 1:
        category = sys.argv[1]
        print(f"Analyzing results for category: {category}")
        results_dir = f"{base_dir}results"
        output_dir = f"{base_dir}results/analysis/{category}"
        
        # Load only the results for the specified category
        results = {}
        methods = ["prompt_optimization", "few_shot", "post_processing", "ensemble"]
        
        for method in methods:
            file_path = os.path.join(results_dir, f"{category}_{method}_results.jsonl")
            if os.path.exists(file_path):
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                results[method] = data
    else:
        results_dir = f"{base_dir}results"
        output_dir = f"{base_dir}results/analysis"
        
        print("Loading results...")
        results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print("Extracting demographic predictions...")
    demographic_predictions = extract_demographic_predictions(results)
    
    print("Calculating demographic parity...")
    demographic_parity = calculate_demographic_parity(demographic_predictions)
    
    print("Calculating ambiguous context metrics...")
    ambiguous_metrics = calculate_ambiguous_context_metrics(demographic_predictions)
    
    print("Calculating equalized odds...")
    equalized_odds = calculate_equalized_odds(demographic_predictions)
    
    print("Calculating intersectional fairness...")
    intersectional_metrics = calculate_intersectional_fairness(demographic_predictions)
    
    print("Calculating accuracy...")
    accuracy = calculate_accuracy(demographic_predictions)
    
    print("Visualizing results...")
    visualize_results(demographic_parity, ambiguous_metrics, equalized_odds, intersectional_metrics, accuracy, output_dir)

if __name__ == "__main__":
    main()
