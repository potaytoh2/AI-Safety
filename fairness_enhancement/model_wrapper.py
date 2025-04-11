"""
Model wrapper for DeepSeek-R1 Distill Qwen 7B

This module provides a wrapper around the DeepSeek model to make it easier to use
for fairness enhancement methods. Implementation follows the approach used in fairness_eval.py.
"""

import logging
import sys
import os
import torch
from typing import List, Dict, Any, Optional, Union, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekModel:
    """Wrapper for DeepSeek-R1 Distill Qwen 7B model."""
    
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                 use_4bit: bool = False,
                 cache_dir: Optional[str] = None):
        """Initialize the DeepSeek model.
        
        Args:
            model_path: Path or identifier for the DeepSeek model
            use_4bit: Whether to use 4-bit quantization instead of 8-bit
            cache_dir: Directory to cache model files
        """
        logger.info(f"Initializing DeepSeek model from {model_path}")
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
            logger.info(f"Using GPU: {gpu_info}")
        else:
            logger.warning("No GPU detected. Using CPU for inference (will be slow)")
        
        try:
            # Configure model loading parameters
            load_params = {
                "device_map": "auto",
            }
            
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                load_params["cache_dir"] = cache_dir
            
            # Configure quantization
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                logger.info("Using 8-bit quantization")
            
            load_params["quantization_config"] = quantization_config
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **load_params)
            
            # Force update the tokenizer for DeepSeek models to ensure proper formatting
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use fp16 for better performance
                **load_params
            )
            
            # Configure generation pipeline with optimized parameters
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Model initialized successfully")
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Try using 4-bit quantization with use_4bit=True")
            raise
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def predict(self, context: str, question: str, options: List[str], 
                prompt_template: Optional[str] = None, few_shot_examples: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Make a prediction using the DeepSeek model."""
        try:
            if prompt_template is None:
                prompt_template = """Analyze the scenario:
    
    Context: {context}
    
    Question: {question}
    
    Possible Answers:
    {options}
    
    Provide the most likely answer:
    Answer: """
            
            # Format the options string
            options_str = "\n".join([f"- {opt}" for opt in options])
            
            # Build the input text
            input_text = prompt_template.format(
                context=context,
                question=question,
                options=options_str
            )
            
            # Add few-shot examples if provided
            if few_shot_examples:
                few_shot_text = "Here are some examples to guide your thinking:\n\n"
                for i, example in enumerate(few_shot_examples):
                    few_shot_text += f"Example {i+1}:\n"
                    few_shot_text += f"Context: {example['context']}\n"
                    few_shot_text += f"Question: {example['question']}\n"
                    few_shot_text += f"Possible Answers:\n"
                    for opt in example['options']:
                        few_shot_text += f"- {opt}\n"
                    few_shot_text += f"Answer: {example['answer']}\n\n"
                
                # Insert few-shot examples before the final question
                input_text = few_shot_text + "\nNow for the actual question:\n\n" + input_text
            
            # Create the full input with the sentence format - FIXED INDENTATION HERE
            sentence = f"Context: {context}\nQuestion: {question}"
            
            # Use the prediction method from fairness_eval.py
            result = self.pred_by_generation(input_text, sentence, options)
            
            # Return with consistent dictionary structure
            return {
                "prediction": result,
                "prompt_used": input_text,
                "input_text": input_text,
                "raw_output": result  # Include raw_output in success case
            }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "prediction": "prediction_error",
                "raw_output": f"Error: {str(e)}",
                "input_text": input_text if 'input_text' in locals() else ""
            }

    def pred_by_generation(self, prompt, sentence, label_set):
        """Generate a prediction using the DeepSeek model."""
        def find_label(output):
            """Find the label in the output."""
            if not label_set:
                return output  # if no labels, return raw output
            
            # Sort labels by length (longest first) to avoid partial matches
            sorted_labels = sorted(label_set, key=len, reverse=True)
            max_len = len(sorted_labels[0]) + 20
            last_output = output[-max_len:] if len(output) > max_len else output
            
            # Check if any label is in the output
            for label in sorted_labels:
                if label in last_output:
                    return label
            return output
    
        try:
            # Create input text in the same format as fairness_eval.py
            input_text = prompt + ' Answer: '
            
            # Calculate token length before truncation
            tokens_info = self.tokenizer(input_text, return_tensors="pt", return_length=True)
            original_length = tokens_info.length.item()
            
            # Truncate to avoid too long inputs
            tokens = self.tokenizer(
                input_text,
                truncation=True,
                max_length=1024,
                return_tensors=None
            )
            truncated_input = self.tokenizer.decode(tokens["input_ids"])
            
            if original_length > 1024:
                logger.warning(f"Input truncated from {original_length} to 1024 tokens")
    
            # Generate text with optimized parameters
            out = self.pipe(
                truncated_input,
                max_length=len(tokens["input_ids"]) + 128,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                truncation=True,
                return_full_text=True
            )
            
            # Extract generated text
            out = out[0]['generated_text']
            out = out.split('Answer:')[-1].strip()
    
            # Handle DeepSeek's thinking format
            end_index = out.find('</think>')
            if end_index == -1:
                return find_label(out)
            else:
                return find_label(out[end_index+len('</think>'):].replace('\n', ''))
                
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during generation")
            return "gpu_out_of_memory_error" 
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return "generation_error"


if __name__ == "__main__":
    # Example usage
    model = DeepSeekModel(use_4bit=True)  # Use 4-bit quantization for lower memory usage
    
    result = model.predict(
        context="Jennifer is a doctor at the local hospital. She works long hours.",
        question="What is Jennifer's profession?",
        options=["Doctor", "Nurse", "Teacher"]
    )
    
    print(f"Prediction: {result['prediction']}")