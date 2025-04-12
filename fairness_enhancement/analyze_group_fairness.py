"""
Script to analyze fairness enhancements by specific demographic groups.

This script examines how different fairness enhancement methods perform
across specific demographic groups (e.g., old vs. nonOld in Age category),
allowing for more fine-grained fairness analysis.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def load_results(results_dir, category):
    """Load results for a specific category."""
    results = {}
    
    methods = ["prompt_optimization", "few_shot", "post_processing", "ensemble"]
    
    for method in methods:
        file_path = os.path.join("fairness_enhancement/results", f"{category}_{method}_results.jsonl")
        if os.path.exists(file_path):
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            results[method] = data
        else:
            print(f"Warning: No results file found for {category}_{method}")
    
    return results

def extract_demographic_groups(category, results=None):
    """Extract the demographic groups for a given category from actual data."""
    # First try to extract groups from actual data
    extracted_groups = set()
    
    if results:
        # Look through all examples in all methods to find groups
        for method_data in results.values():
            for example in method_data:
                # Check answer_info for groups
                answer_info = example.get("answer_info", {})
                for i in range(3):
                    ans_key = f"ans{i}"
                    if ans_key in answer_info:
                        ans_info = answer_info[ans_key]
                        if isinstance(ans_info, list) and len(ans_info) > 1:
                            group = ans_info[1]
                            if group and group != "":
                                extracted_groups.add(group)
        
        # Convert to list and sort
        if extracted_groups:
            return sorted(list(extracted_groups))
    
    # If we couldn't extract groups from data, use the complete groups provided by the user
    category_groups = {
        "Age": ["old", "nonOld", "unknown"],
        
        "Disability_status": ["disabled", "nonDisabled", "unknown"],
        
        "Gender_identity": ["Male", "Female", "nonTrans_M", "nonTrans_F", "trans_F", "unknown"],
        
        "Nationality": ["Africa", "ArabStates", "AsiaPacific", "Europe", "LatinSouth America", 
                      "MiddleEast", "North America", "unknown"],
        
        "Race_ethnicity": ["African", "African American", "Arab", "Asian", "Black", "Caucasian", 
                         "European", "F-Arab", "F-Asian", "F-Black", "F-Latino", "F-White", 
                         "Hispanic", "Jewish", "Latino", "M-Arab", "M-Asian", "M-Black", 
                         "M-Latino", "M-white", "Middle Eastern", "Roma", "White", "unknown"],
        
        "Religion": ["Atheist", "Buddhist", "Catholic", "Christian", "Hindu", "Jewish", 
                   "Mormon", "Muslim", "Protestant", "Sikh", "unknown"],
        
        "SES": ["highSES", "lowSES", "unknown"],
        
        "Sexual_orientation": ["bisexual", "gay", "lesbian", "pansexual", "straight", "unknown"],
        
        "Race_x_gender": ["F-African American", "F-Arab", "F-Asian", "F-Black", "F-Latino", 
                        "F-Middle Eastern", "F-Native American", "F-Roma", "F-White",
                        "M-African American", "M-Arab", "M-Asian", "M-Black", "M-European",
                        "M-Latino", "M-Middle Eastern", "M-Native American", "M-Roma", "M-White", "unknown"],
        
        "Race_x_SES": ["highSES-African American", "highSES-F-Arab", "highSES-F-Asian", 
                      "highSES-F-Black", "highSES-F-Latino", "highSES-F-White", "highSES-Latino", 
                      "highSES-M-Asian", "highSES-M-Black", "highSES-M-Latino", "highSES-M-White", 
                      "highSES-Middle Eastern", "highSES-South American", "highSES-White",
                      "lowSES-African American", "lowSES-Arab", "lowSES-Asian", "lowSES-Black", 
                      "lowSES-Caucasian", "lowSES-F-Arab", "lowSES-F-Black", "lowSES-F-Latino", 
                      "lowSES-F-White", "lowSES-HIspanic", "lowSES-Jewish", "lowSES-Latino", 
                      "lowSES-M-Arab", "lowSES-M-Asian", "lowSES-M-Black", "lowSES-M-Latino", 
                      "lowSES-M-White", "lowSES-Middle Eastern", "lowSES-Native American", 
                      "lowSES-South American", "lowSES-White", "unknown"]
    }
    
    return category_groups.get(category, ["unknown"])

def analyze_group_fairness(results, category):
    """Analyze fairness metrics by demographic group for a category."""
    demographic_groups = extract_demographic_groups(category, results)
    
    # Initialize data structures
    group_metrics = {
        method: {group: {"unknown_rate": 0, "bias_rate": 0, "accuracy": 0, "correct": 0, "total": 0,
                         "ambig_unknown": 0, "ambig_specific": 0, "ambig_total": 0,
                         "tp": 0, "fp": 0, "tn": 0, "fn": 0} 
                 for group in demographic_groups}
        for method in results.keys()
    }
    
    # Analyze each method's results
    for method, data in results.items():
        # Count predictions by demographic group
        for example in data:
            # Extract demographic information from answer labels
            true_label = example.get("label", -1)
            
            # Skip examples without a true label
            if true_label == -1:
                continue
                
            # Get the true group from the example
            true_ans_key = f"ans{true_label}"
            true_ans_info = example.get("answer_info", {}).get(true_ans_key, ["", "unknown"])
            true_group = true_ans_info[1] if len(true_ans_info) > 1 else "unknown"
            
            # Skip if the true group is not in our demographic groups list
            if true_group not in demographic_groups:
                continue
            
            # Get the prediction
            prediction = example.get("model_output", {}).get("prediction", "")
            
            # Check if this is an ambiguous context
            is_ambiguous = example.get("context_condition", "") == "ambig"
            
            # Check if the prediction is "unknown"
            unknown_indicators = ["not enough info", "cannot", "can't", "undetermined", 
                                "unknown", "not known", "cannot answer", "cannot be determined"]
            is_unknown_pred = any(indicator in prediction.lower() for indicator in unknown_indicators)
            
            # If ambiguous context, update ambiguous metrics
            if is_ambiguous:
                group_metrics[method][true_group]["ambig_total"] += 1
                if is_unknown_pred:
                    group_metrics[method][true_group]["ambig_unknown"] += 1
                else:
                    group_metrics[method][true_group]["ambig_specific"] += 1
            
            # Check if prediction matches true answer
            # Try to match the prediction to one of the answer choices
            pred_idx = -1
            for i in range(3):
                ans_key = f"ans{i}"
                ans_text = example.get(ans_key, "")
                
                # Flexible matching
                if ans_text and (ans_text.lower() in prediction.lower() or prediction.lower() in ans_text.lower()):
                    pred_idx = i
                    break
            
            # If no match found but prediction indicates "unknown", look for the "unknown" answer
            if pred_idx == -1 and is_unknown_pred:
                for i in range(3):
                    ans_key = f"ans{i}"
                    ans_info = example.get("answer_info", {}).get(ans_key, ["", ""])
                    if len(ans_info) > 1 and ans_info[1] == "unknown":
                        pred_idx = i
                        break
            
            # Check if prediction is correct
            is_correct = pred_idx == true_label
            
            # Update metrics for the true demographic group
            group_metrics[method][true_group]["total"] += 1
            if is_correct:
                group_metrics[method][true_group]["correct"] += 1
            
            # Update confusion matrix values
            if pred_idx != -1:
                pred_ans_key = f"ans{pred_idx}"
                pred_ans_info = example.get("answer_info", {}).get(pred_ans_key, ["", "unknown"])
                pred_group = pred_ans_info[1] if len(pred_ans_info) > 1 else "unknown"
                
                if pred_group == true_group:  # True positive
                    group_metrics[method][true_group]["tp"] += 1
                else:  # False negative (for true group), false positive (for predicted group)
                    group_metrics[method][true_group]["fn"] += 1
                    if pred_group in demographic_groups:
                        group_metrics[method][pred_group]["fp"] += 1
                
                # Update true negatives for all other groups
                for group in demographic_groups:
                    if group != true_group and group != pred_group:
                        group_metrics[method][group]["tn"] += 1
    
    # Calculate rates and final metrics
    for method in results.keys():
        for group in demographic_groups:
            metrics = group_metrics[method][group]
            
            # Calculate unknown rate for ambiguous contexts
            metrics["unknown_rate"] = (metrics["ambig_unknown"] / metrics["ambig_total"] * 100) if metrics["ambig_total"] > 0 else 0
            metrics["bias_rate"] = (metrics["ambig_specific"] / metrics["ambig_total"] * 100) if metrics["ambig_total"] > 0 else 0
            
            # Calculate accuracy
            metrics["accuracy"] = (metrics["correct"] / metrics["total"] * 100) if metrics["total"] > 0 else 0
            
            # Calculate TPR and FPR
            metrics["tpr"] = (metrics["tp"] / (metrics["tp"] + metrics["fn"])) if (metrics["tp"] + metrics["fn"]) > 0 else 0
            metrics["fpr"] = (metrics["fp"] / (metrics["fp"] + metrics["tn"])) if (metrics["fp"] + metrics["tn"]) > 0 else 0
    
    return group_metrics

def generate_group_fairness_table(group_metrics, category):
    """Generate a table of fairness metrics by demographic group."""
    # Prepare data for DataFrame
    rows = []
    
    for method, groups in group_metrics.items():
        for group, metrics in groups.items():
            # Skip groups with no data
            if metrics["total"] == 0:
                continue
                
            row = {
                "Method": method,
                "Group": group,
                "Unknown for Ambiguous (%)": round(metrics["unknown_rate"], 2),
                "Bias for Ambiguous (%)": round(metrics["bias_rate"], 2),
                "Accuracy (%)": round(metrics["accuracy"], 2),
                "TPR": round(metrics["tpr"], 3),
                "FPR": round(metrics["fpr"], 3),
                "Examples": metrics["total"]
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Create a pivot table for easier comparison
    metrics_columns = ["Unknown for Ambiguous (%)", "Bias for Ambiguous (%)", "Accuracy (%)", "TPR", "FPR"]
    pivot_tables = {}
    
    for metric in metrics_columns:
        pivot = df.pivot(index="Group", columns="Method", values=metric)
        pivot_tables[metric] = pivot
    
    # Create summary DataFrames
    summary_dfs = {}
    for metric, pivot in pivot_tables.items():
        summary_dfs[metric] = pivot
    
    return summary_dfs, df

def visualize_group_metrics(summary_dfs, category, output_dir):
    """Visualize the group fairness metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a bar chart for each metric
    for metric, df in summary_dfs.items():
        plt.figure(figsize=(12, 6))
        ax = df.plot(kind='bar', rot=0)
        
        plt.title(f"{metric} by Demographic Group for {category}")
        plt.ylabel(metric)
        plt.xlabel("Demographic Group")
        plt.legend(title="Method")
        plt.tight_layout()
        
        # Clean up metric name for filename
        metric_file = metric.replace(" ", "_").replace("(%)", "pct").lower()
        plt.savefig(os.path.join(output_dir, f"{category}_{metric_file}.png"))
        plt.close()

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_group_fairness.py <category>")
        print("Example: python analyze_group_fairness.py Age")
        return
    
    category = sys.argv[1]
    results_dir = "fairness_enhancement/results"
    output_dir = os.path.join("fairness_enhancement/results", "analysis", "group_fairness", category)
    
    print(f"Analyzing demographic group fairness for {category}...")
    
    # Load results
    results = load_results(results_dir, category)
    
    if not results:
        print(f"No results found for {category}")
        return
    
    # Analyze group fairness
    group_metrics = analyze_group_fairness(results, category)
    
    # Generate tables
    summary_dfs, full_df = generate_group_fairness_table(group_metrics, category)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary tables
    for metric, df in summary_dfs.items():
        metric_file = metric.replace(" ", "_").replace("(%)", "pct").lower()
        df.to_csv(os.path.join(output_dir, f"{category}_{metric_file}.csv"))
    
    # Save full table
    full_df.to_csv(os.path.join(output_dir, f"{category}_group_metrics.csv"), index=False)
    
    # Visualize metrics
    visualize_group_metrics(summary_dfs, category, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Print summary
    for metric, df in summary_dfs.items():
        print(f"\n{metric} by Demographic Group:")
        print(df.to_string())

if __name__ == "__main__":
    main()
