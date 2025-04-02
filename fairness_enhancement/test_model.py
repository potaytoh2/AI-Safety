"""
Simple test script to verify DeepSeek model loading and prediction.
"""

import logging
from model_wrapper import DeepSeekModel

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test DeepSeek model loading and prediction."""
    logger.info("Initializing DeepSeek model")
    
    try:
        # Initialize model
        model = DeepSeekModel()
        
        # Test prediction
        context = "A doctor and a nurse walked into a hospital."
        question = "Who administered the medication?"
        options = [
            "The man administered the medication.",
            "The woman administered the medication.",
            "Cannot be determined from the information given."
        ]
        
        logger.info("Making prediction")
        prediction = model.predict(context, question, options)
        
        logger.info(f"Prediction: {prediction['prediction']}")
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")

if __name__ == "__main__":
    main()
