# Fairness Enhancement for DeepSeek-R1 Distill Qwen 7B

This module implements fairness enhancement methods for the DeepSeek-R1 Distill Qwen 7B model as described in the project proposal. The implementation provides three distinct methods to improve fairness in the model's outputs, plus an ensemble approach that combines these methods.

## Overview

The fairness enhancement methods are designed to mitigate biases against protected demographic groups in the model's outputs. The implementation follows the approach outlined in the project proposal, including:

1. **Fair Prompt Optimization** - Iteratively refines prompts to promote fairness in model responses.
2. **Strategic Few-Shot Selection** - Carefully curates diverse examples for in-context learning to improve fairness.
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
├── README.md                # This documentation
└── requirements.txt         # Dependencies
```

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
  --output_file fairness_enhancement/results/gender_identity_prompt_optimized.jsonl \
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

### Example: Running Ensemble Method

```bash
python fairness_enhancement/main.py \
  --input_file fairness/Bias-Benchmark/data/Race_ethnicity.jsonl \
  --output_file fairness_enhancement/results/race_ethnicity_ensemble.jsonl \
  --method ensemble \
  --category Race_ethnicity
```

## Method Explanations

### 1. Fair Prompt Optimization

This method iteratively optimizes prompts to enhance fairness in model outputs. It:

1. Evaluates baseline performance on different demographic groups
2. Identifies problematic groups with the largest fairness disparities
3. Optimizes prompts to reduce bias against these groups
4. Applies the optimized prompt to all examples

The implementation is based on the approach described in:
> Cherepanova, V., Lee, C.-J., Akpinar, N., Fogliato, R., Bertran, M. A., Kearns, M., & Zou, J. (2024). Improving LLM group fairness on tabular data via in-context learning.

### 2. Strategic Few-Shot Selection

This method strategically selects few-shot examples to provide to the model for in-context learning. It:

1. Creates pools of examples for each demographic group
2. Identifies challenging examples where the model might show bias
3. Selects few-shot examples based on:
   - Including examples from underrepresented groups
   - Including examples similar to the current example
   - Including challenging examples
4. Applies the selected few-shot examples to enhance fairness

The implementation is based on the approach described in:
> Hu, J., Liu, W., & Du, M. (2024). Strategic demonstration selection for improved fairness in LLM in-context learning.

### 3. Post-Processing Techniques

This method applies post-processing corrections to the model's outputs without modifying the model itself. It:

1. Splits data into calibration and evaluation sets
2. Trains demographic parity correction factors to balance prediction rates
3. Trains a logistic regression model to predict correct answers
4. Applies both correction methods and selects the best one based on the context

The implementation is based on the approach described in:
> Kadhe, S. R., Halimi, A., Rawat, A., & Baracaldo, N. (2023). FairSISA: Ensemble post-processing to improve fairness of unlearning in LLMs.

### 4. Fairness Ensemble

This method combines all three fairness enhancement approaches to achieve comprehensive bias mitigation. It supports three voting strategies:

1. **Majority Voting**: Simple majority vote from all methods
2. **Weighted Voting**: Weights each method based on its performance on calibration data
3. **Contextual Voting**: Selects the most appropriate method based on the example's properties

The ensemble method dynamically adjusts weights based on each method's accuracy and fairness metrics on calibration data.

## Evaluation

After applying fairness enhancement, you can evaluate the enhanced outputs using the existing fairness evaluation metrics in `fairness/metrics.ipynb`. This will allow you to compare:

- Demographic Parity before and after enhancement
- Equalized Odds (TPR, FPR) before and after enhancement
- Overall accuracy and group-specific accuracy

## References

1. Cherepanova, V., Lee, C.-J., Akpinar, N., Fogliato, R., Bertran, M. A., Kearns, M., & Zou, J. (2024). Improving LLM group fairness on tabular data via in-context learning.
2. Hu, J., Liu, W., & Du, M. (2024). Strategic demonstration selection for improved fairness in LLM in-context learning.
3. Kadhe, S. R., Halimi, A., Rawat, A., & Baracaldo, N. (2023). FairSISA: Ensemble post-processing to improve fairness of unlearning in LLMs.
4. Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A hand-built bias benchmark for question answering.
