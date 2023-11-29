import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

class TestCode(unittest.TestCase):

    # Set up necessary resources and configurations for tests
    def setUp(self):
        self.model_directory = "Software"
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        
        # Ensure model is on the appropriate device (CPU/GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Create a text generation pipeline
        self.text_generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, device=0, max_length=2000)

    # Test that the model and tokenizer are loaded successfully    
    def test_model__and_tokenizer_loading(self):
        self.assertIsInstance(self.model, AutoModelForCausalLM)
        self.assertIsInstance(self.tokenizer, AutoTokenizer)
    
     # Test that the model is on the correct device (GPU/CPU)
    def test_gpu_usage(self):
        self.assertEqual(self.device, "cuda" if torch.cuda.is_available() else "cpu")
    
    # Test that the text generation pipeline is created successfully
    def test_pipeline_creation(self):
        self.assertIsNotNone(self.text_generation_pipeline)
    
    # Test the generate_text function with a simple prompt
    def test_text_generation_function(self):
        prompt = "This is a test prompt"
        generated_text = generate_text(prompt)
        self.assertTrue(isinstance(generated_text, str) and len(generated_text) > 0)

    # Test the model against a longer input
    def test_generate_text_long_input(self):
        prompt = "This is a very long prompt. " * 50
        max_token_length = 2000

        #Check for raised error
        with self.assertRaises(ValueError):
            generate_text(prompt, max_length=max_length)

if __name__ == '__main__':
    unittest.main()