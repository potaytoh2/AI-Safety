"""
Run group fairness analysis for all demographic categories.

This script automatically runs the analyze_group_fairness.py script
for all demographic categories in the dataset, generating detailed
fairness metrics broken down by specific demographic groups within
each category.
"""

import os
import subprocess
import sys
import pandas as pd
from collections import defaultdict

# Define the categories to analyze
CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
    "Sexual_orientation"
]

def run_analysis(categories=None):
    """Run group fairness analysis for specified categories."""
    if categories is None:
        categories = CATEGORIES
    
    results_dir = os.path.join("fairness_enhancement/results", "analysis", "group_fairness")
    os.makedirs(results_dir, exist_ok=True)
    
    # Store summary results for all categories
    all_results = defaultdict(list)
    
    for category in categories:
        print(f"\n{'='*80}")
        print(f"Analyzing {category}...")
        print(f"{'='*80}")
        
        # Run analyze_group_fairness.py for this category
        cmd = ["python", "fairness_enhancement/analyze_group_fairness.py", category]
        subprocess.run(cmd)
        
        # Load results if available
        category_dir = os.path.join(results_dir, category)
        unknown_rate_file = os.path.join(category_dir, f"{category}_unknown_for_ambiguous_pct.csv")
        
        if os.path.exists(unknown_rate_file):
            # Get "Unknown for Ambiguous (%)" metric as our primary fairness metric
            unknown_rates = pd.read_csv(unknown_rate_file)
            
            # For each group and method in this category
            for group in unknown_rates.index:
                for method in unknown_rates.columns:
                    value = unknown_rates.loc[group, method]
                    all_results["Category"].append(category)
                    all_results["Group"].append(group)
                    all_results["Method"].append(method)
                    all_results["Unknown Rate (%)"].append(value)
    
    # Create a consolidated summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(results_dir, "all_group_metrics.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nConsolidated results saved to {summary_file}")
        
        # Make sure the Unknown Rate is numeric and create a cleaner view
        summary_df["Unknown Rate (%)"] = pd.to_numeric(summary_df["Unknown Rate (%)"], errors='coerce')
        
        # Create a more readable pivot view without using pivot_table
        pivot_view = summary_df.drop_duplicates(['Category', 'Group', 'Method'])
        pivot_view = pivot_view.pivot(index=['Category', 'Group'], 
                                     columns='Method', 
                                     values='Unknown Rate (%)')
        pivot_file = os.path.join(results_dir, "group_unknown_rates_by_method.csv")
        pivot_view.to_csv(pivot_file)
        print(f"Pivot view saved to {pivot_file}")
        
        # Convert to numeric to find max rates
        summary_df["Unknown Rate (%)"] = pd.to_numeric(summary_df["Unknown Rate (%)"], errors='coerce')
        
        # Print summary of best methods per group
        best_method_indices = summary_df.groupby(['Category', 'Group'])['Unknown Rate (%)'].idxmax()
        best_method_df = summary_df.loc[best_method_indices]
        best_method_df = best_method_df[['Category', 'Group', 'Method', 'Unknown Rate (%)']]
        best_method_file = os.path.join(results_dir, "best_methods_by_group.csv")
        best_method_df.to_csv(best_method_file, index=False)
        print(f"Best methods per group saved to {best_method_file}")
        
        # Print a summary of results
        print("\nSummary of Best Methods per Demographic Group:")
        print("------------------------------------------------")
        
        for category in CATEGORIES:
            category_results = best_method_df[best_method_df['Category'] == category]
            if not category_results.empty:
                print(f"\n{category}:")
                for _, row in category_results.iterrows():
                    print(f"  {row['Group']}: {row['Method']} ({row['Unknown Rate (%)']:.2f}%)")

if __name__ == "__main__":
    # Allow user to specify categories via command line
    if len(sys.argv) > 1:
        categories = sys.argv[1:]
        run_analysis(categories)
    else:
        # Run all categories
        run_analysis()
