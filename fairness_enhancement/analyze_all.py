"""
Run analysis for all categories and create a combined summary based on
the fairness metrics specified in the CS427 project proposal.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    # List of all categories as specified in the project
    categories = [
        "Age", "Disability_status", "Gender_identity", "Nationality",
        "Physical_appearance", "Race_ethnicity", "Race_x_gender",
        "Race_x_SES", "Religion", "SES", "Sexual_orientation"
    ]
    
    print(f"Running analysis for {len(categories)} categories...")
    
    # Run analysis for each category
    for category in categories:
        print(f"\nProcessing {category}...")
        subprocess.run(["python", "analyze_results.py", category], check=True)
    
    # Combine all summary.csv files into one
    combined_data = []
    
    for category in categories:
        summary_path = os.path.join("results", "analysis", category, "summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df["Category"] = category
            combined_data.append(df)
    
    if combined_data:
        # Combine all dataframes
        combined_df = pd.concat(combined_data)
        
        # Reorder columns to put Category first
        cols = combined_df.columns.tolist()
        cols.remove("Category")
        cols = ["Category"] + cols
        combined_df = combined_df[cols]
        
        # Save combined summary
        analysis_dir = os.path.join("results", "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        combined_path = os.path.join(analysis_dir, "combined_summary.csv")
        combined_df.to_csv(combined_path, index=False)
        
        # Print statistics
        print("\nOverall Averages by Method:")
        # Filter only numeric columns for averaging
        numeric_columns = combined_df.select_dtypes(include=['number']).columns.tolist()
        method_averages = combined_df.groupby("Method")[numeric_columns].mean().reset_index()
        print(method_averages.to_string(index=False))
        
        # Create summary visualizations
        create_summary_visualizations(combined_df, analysis_dir)
        
        print("\nCombined summary saved to:", combined_path)


def create_summary_visualizations(combined_df, output_dir):
    """Create summary visualizations across all categories."""
    methods = combined_df["Method"].unique()
    categories = combined_df["Category"].unique()
    
    # 1. Average Unknown for Ambiguous by Method
    plt.figure(figsize=(12, 6))
    avg_by_method = combined_df.groupby("Method")["Unknown for Ambiguous (%)"].mean().reset_index()
    plt.bar(avg_by_method["Method"], avg_by_method["Unknown for Ambiguous (%)"])
    plt.title("Average 'Unknown' Response Rate for Ambiguous Contexts by Method")
    plt.xlabel("Method")
    plt.ylabel("Average Unknown Response Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_unknown_by_method.png"))
    
    # 2. Average Accuracy by Method
    plt.figure(figsize=(12, 6))
    avg_accuracy = combined_df.groupby("Method")["Accuracy (%)"].mean().reset_index()
    plt.bar(avg_accuracy["Method"], avg_accuracy["Accuracy (%)"])
    plt.title("Average Accuracy by Fairness Enhancement Method")
    plt.xlabel("Method")
    plt.ylabel("Average Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_accuracy_by_method.png"))
    
    # 3. Average Fairness Disparities by Method
    plt.figure(figsize=(12, 8))
    
    # Select disparity metrics
    disparity_metrics = ["TPR Disparity", "FPR Disparity", "Accuracy Disparity (%)"]
    avg_disparities = combined_df.groupby("Method")[disparity_metrics].mean().reset_index()
    
    # Set up plot
    x = np.arange(len(methods))
    width = 0.25
    
    # Plot each disparity metric as a group of bars
    plt.bar(x - width, avg_disparities["TPR Disparity"], width, label="TPR Disparity")
    plt.bar(x, avg_disparities["FPR Disparity"], width, label="FPR Disparity")
    plt.bar(x + width, avg_disparities["Accuracy Disparity (%)"], width, label="Accuracy Disparity (%)")
    
    plt.title("Average Fairness Disparities by Method (Lower is Better)")
    plt.xlabel("Method")
    plt.ylabel("Average Disparity")
    plt.xticks(x, methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_disparities_by_method.png"))
    
    # 4. Heatmap of Unknown Response Rate by Category and Method
    plt.figure(figsize=(14, 10))
    pivot_df = combined_df.pivot(index="Category", columns="Method", values="Unknown for Ambiguous (%)")
    
    # Create heatmap
    plt.imshow(pivot_df.values, cmap="viridis", aspect="auto")
    plt.colorbar(label="Unknown Response Rate (%)")
    
    # Add labels
    plt.xticks(np.arange(len(methods)), methods, rotation=45)
    plt.yticks(np.arange(len(categories)), categories)
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(methods)):
            value = pivot_df.iloc[i, j]
            color = "white" if value > 50 else "black"
            plt.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color)
    
    plt.title("'Unknown' Response Rate for Ambiguous Contexts by Category and Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unknown_rate_heatmap.png"))
    
    # 5. Method Performance Ranking
    # Create a score for each method (higher is better)
    method_scores = defaultdict(float)
    
    # Weights for different metrics
    weights = {
        "Unknown for Ambiguous (%)": 1.0,      # Higher is better
        "Bias for Ambiguous (%)": -1.0,        # Lower is better
        "Accuracy (%)": 0.5,                   # Higher is better
        "TPR Disparity": -0.5,                 # Lower is better
        "FPR Disparity": -0.5,                 # Lower is better
        "Accuracy Disparity (%)": -0.7         # Lower is better
    }
    
    # Normalize each metric to [0, 1] range within each category
    for category in categories:
        category_df = combined_df[combined_df["Category"] == category].copy()
        
        for metric, weight in weights.items():
            if metric not in category_df.columns:
                continue
                
            # Skip this metric if any non-numeric values
            if not pd.api.types.is_numeric_dtype(category_df[metric]):
                continue
                
            values = category_df[metric].values
            if len(values) == 0:
                continue
                
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val
            
            if range_val == 0:  # All methods have the same value
                continue
                
            for method in category_df["Method"].unique():
                method_rows = category_df["Method"] == method
                if not any(method_rows):
                    continue
                    
                method_value = category_df.loc[method_rows, metric].iloc[0]
                normalized_value = (method_value - min_val) / range_val
                
                # For metrics where higher is better, keep normalized value
                # For metrics where lower is better, invert normalized value
                if weight < 0:
                    normalized_value = 1 - normalized_value
                    
                method_scores[method] += abs(weight) * normalized_value
    
    # Create ranking visualization
    methods_ranked = sorted(method_scores.keys(), key=lambda m: method_scores[m], reverse=True)
    scores = [method_scores[m] for m in methods_ranked]
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods_ranked, scores)
    plt.title("Overall Fairness Enhancement Method Ranking (Higher is Better)")
    plt.xlabel("Method")
    plt.ylabel("Weighted Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_ranking.png"))
    
    # Save ranking to CSV
    ranking_df = pd.DataFrame({
        "Method": methods_ranked,
        "Score": scores,
        "Rank": range(1, len(methods_ranked) + 1)
    })
    ranking_df.to_csv(os.path.join(output_dir, "method_ranking.csv"), index=False)

if __name__ == "__main__":
    main()
