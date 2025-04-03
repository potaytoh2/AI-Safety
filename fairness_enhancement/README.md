# Fairness Enhancement for DeepSeek-R1 Distill Qwen 7B

This module implements fairness enhancement methods for the DeepSeek-R1 Distill Qwen 7B model to mitigate biases against protected demographic groups in question answering tasks. The implementation provides three distinct approaches to improve fairness in the model's outputs, plus an ensemble method that combines these approaches.

## Overview

The fairness enhancement methods are designed to address biases in the BBQ (Bias Benchmark for Question Answering) dataset. According to our analysis, these enhancements have successfully improved fairness metrics across different bias categories. Results show that prompt optimization is the most effective approach, followed by few-shot selection, while the ensemble and post-processing methods also provide meaningful improvements.

1. **Fair Prompt Optimization** - Iteratively refines prompts to reduce reliance on stereotypes, especially for ambiguous contexts.
2. **Strategic Few-Shot Selection** - Carefully curates diverse examples for in-context learning to improve fairness across demographic groups.
3. **Post-Processing Techniques** - Applies bias correction to model outputs without modifying the model itself.
4. **Fairness Ensemble** - Combines the above methods using weighted voting or contextual selection.

## Directory Structure

```
fairness_enhancement/
├── main.py                  # Main entry point for running fairness enhancement
├── model_wrapper.py         # Wrapper for DeepSeek-R1 model
├── prompt_optimization.py   # Fair prompt optimization implementation
├── few_shot_selection.py    # Strategic few-shot example selection
├── post_processing.py       # Post-processing bias correction
├── ensemble.py              # Ensemble approach combining multiple methods
├── analyze_results.py       # Script to analyze individual category results
├── analyze_all.py           # Script to run analysis across all categories
├── run_all.py               # Script to run all enhancement methods on all categories
├── README.md                # This documentation
└── requirements.txt         # Dependencies
├── results/                 # Enhanced prediction results and analysis
    ├── [category]_[method]_results.jsonl    # Enhancement results by category and method
    ├── samples/                             # Sample data for each category
    └── analysis/                            # Analysis results and visualizations
```

## Performance Results

Based on our analysis (see `results/analysis/method_ranking.csv`), the fairness enhancement methods ranked from most to least effective are:

1. **Prompt Optimization** (Score: 27.47) - Most effective at responding with "unknown" for ambiguous contexts
2. **Few-Shot Selection** (Score: 18.87) - Strong performance on demographic parity
3. **Ensemble** (Score: 7.67) - Balanced approach with moderate improvements
4. **Post-Processing** (Score: 7.15) - Best at reducing specific disparities but with accuracy trade-offs

Key improvements across methods:
- Increased "unknown" responses for ambiguous contexts (reduced stereotyping)
- Reduced disparities in true positive rates (TPR) between demographic groups
- More balanced accuracy across protected groups
- Reduced influence of social biases on model predictions

## Installation

1. Ensure you have the required dependencies:

```bash
pip install -r fairness_enhancement/requirements.txt
```

2. The fairness enhancement module works with the existing fairness evaluation infrastructure. It uses the same BBQ dataset and evaluation framework.

## Usage

### Running a Single Method

To enhance fairness using a single method, use the `main.py` script:

```bash
python fairness_enhancement/main.py \
  --input_file fairness/Bias-Benchmark/data/Gender_identity.jsonl \
  --output_file fairness_enhancement/results/Gender_identity_prompt_optimization_results.jsonl \
  --method prompt_optimization \
  --category Gender_identity
```

### Available Methods

- `prompt_optimization`: Uses optimized prompts to enhance fairness
- `few_shot`: Uses strategic few-shot example selection
- `post_processing`: Applies post-processing bias correction
- `ensemble`: Combines all methods using weighted voting

### Additional Options

- `--category`: Bias category to focus on (e.g., Gender_identity, Race_ethnicity)
- `--meta_llm`: Meta LLM to use for prompt optimization (optional)
- `--num_examples`: Number of few-shot examples to use (default: 5)
- `--model_path`: Path or identifier for the DeepSeek model

### Running All Enhancement Methods

To run all methods on all categories, use the `run_all.py` script:

```bash
python fairness_enhancement/run_all.py
```

### Analyzing Results

After running enhancement methods, analyze their effectiveness:

```bash
# Analyze a specific category
python fairness_enhancement/analyze_results.py Gender_identity

# Analyze all categories and create summary
python fairness_enhancement/analyze_all.py
```

## Method Implementations

### 1. Fair Prompt Optimization (`prompt_optimization.py`)

This method iteratively optimizes prompts to enhance fairness in model outputs. It:

1. Evaluates baseline performance on different demographic groups
2. Identifies problematic groups with the largest fairness disparities
3. Uses a set of fairness-oriented templates that:
   - Emphasize objectivity and impartiality
   - Explicitly warn against demographic stereotypes
   - Encourage "unknown" responses when information is ambiguous
4. Applies the optimized prompt to all examples

The implementation shows strong performance in handling ambiguous contexts, with the highest overall fairness score across categories.

Reference:
> Cherepanova, V., Lee, C.-J., Akpinar, N., Fogliato, R., Bertran, M. A., Kearns, M., & Zou, J. (2024). Improving LLM group fairness on tabular data via in-context learning.

### 2. Strategic Few-Shot Selection (`few_shot_selection.py`)

This method strategically selects few-shot examples to provide to the model for in-context learning. It:

1. Creates pools of examples for each demographic group
2. Identifies challenging examples where the model might show bias
3. Selects few-shot examples based on:
   - Including examples from underrepresented groups
   - Including examples similar to the current example
   - Including challenging examples
4. Applies the selected few-shot examples to enhance fairness

This approach performs particularly well on race and gender categories, showing strong improvements in fairness metrics while maintaining good accuracy.

Reference:
> Hu, J., Liu, W., & Du, M. (2024). Strategic demonstration selection for improved fairness in LLM in-context learning.

### 3. Post-Processing Techniques (`post_processing.py`)

This method applies post-processing corrections to the model's outputs without modifying the model itself. It:

1. Splits data into calibration and evaluation sets
2. Trains demographic parity correction factors to balance prediction rates across groups
3. Trains a logistic regression model to predict correct answers based on model outputs and demographic features
4. Applies both correction methods and selects the best one based on the context:
   - For ambiguous contexts: prefer demographic parity to avoid bias
   - For disambiguated contexts: prefer logistic correction for accuracy

While this method shows the lowest overall score, it provides the most consistent bias reduction in specific categories.

Reference:
> Kadhe, S. R., Halimi, A., Rawat, A., & Baracaldo, N. (2023). FairSISA: Ensemble post-processing to improve fairness of unlearning in LLMs.

### 4. Fairness Ensemble (`ensemble.py`)

This method combines all three fairness enhancement approaches to achieve comprehensive bias mitigation. It supports three voting strategies:

1. **Majority Voting**: Simple majority vote from all methods
2. **Weighted Voting**: Weights each method based on its performance on calibration data
3. **Contextual Voting**: Selects the most appropriate method based on the example's properties:
   - For ambiguous contexts: prioritizes prompt optimization
   - For disambiguated contexts: prioritizes few-shot selection or post-processing

The ensemble approach provides balanced results across different fairness metrics but may not always outperform the best individual method for each specific category.

## Fairness Metrics

The fairness enhancement methods are evaluated on several key metrics:

1. **Unknown for Ambiguous (%)**: The percentage of "unknown" responses for ambiguous contexts. Higher values indicate reduced reliance on stereotypes.

2. **Bias for Ambiguous (%)**: The percentage of specific group predictions for ambiguous contexts. Lower values indicate reduced stereotyping.

3. **TPR Disparity**: The maximum difference in True Positive Rate between demographic groups. Lower values indicate more equal treatment.

4. **FPR Disparity**: The maximum difference in False Positive Rate between demographic groups. Lower values indicate more equal error rates.

5. **Accuracy Disparity (%)**: The maximum difference in accuracy between demographic groups. Lower values indicate more consistent performance.

6. **Overall Accuracy (%)**: The percentage of correct predictions across all examples, measuring general performance.

## Analysis Results

Our analysis in `results/analysis/` shows that:

- **Prompt optimization** achieves the highest "unknown" response rate for ambiguous contexts, demonstrating its effectiveness at reducing stereotyping
- **Few-shot selection** shows the best balance between accuracy and fairness across demographic groups
- **Post-processing** achieves the strongest reduction in specific disparities for some categories
- **Ensemble** provides moderate improvements across multiple metrics simultaneously

Each enhancement method shows strengths in different bias categories, with prompt optimization being particularly effective for Age, Nationality, and Race categories, while few-shot selection excels in Gender_identity and Religion categories.

## References

1. Cherepanova, V., Lee, C.-J., Akpinar, N., Fogliato, R., Bertran, M. A., Kearns, M., & Zou, J. (2024). Improving LLM group fairness on tabular data via in-context learning.
2. Hu, J., Liu, W., & Du, M. (2024). Strategic demonstration selection for improved fairness in LLM in-context learning.
3. Kadhe, S. R., Halimi, A., Rawat, A., & Baracaldo, N. (2023). FairSISA: Ensemble post-processing to improve fairness of unlearning in LLMs.
4. Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A hand-built bias benchmark for question answering.
