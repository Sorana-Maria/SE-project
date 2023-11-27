import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

class TestYourCode(unittest.TestCase):

    def setUp(self):
        # Set up any necessary resources or configurations for tests
        self.model_directory = "Software"
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        
        # Ensure model is on the appropriate device (CPU/GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Create a text generation pipeline
        self.text_generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, device=0, max_length=2000)

    def test_model_loading(self):
        # Test that the model and tokenizer are loaded successfully
        self.assertIsInstance(self.model, AutoModelForCausalLM)
        self.assertIsInstance(self.tokenizer, AutoTokenizer)

    def test_text_generation_function(self):
        # Test the generate_text function with a simple prompt
        prompt = "Test prompt"
        generated_text = generate_text(prompt)
        self.assertTrue(isinstance(generated_text, str) and len(generated_text) > 0)

    def test_gpu_usage(self):
        # Test that the model is on the correct device (GPU/CPU)
        self.assertEqual(self.device, "cuda" if torch.cuda.is_available() else "cpu")

    def test_input_validation_max_token_length(self):
        # Test input validation for max token length
        input_token_length_numeric = "500"
        result_numeric = generate_text("Test prompt", max_length=int(input_token_length_numeric))
        self.assertTrue(isinstance(result_numeric, str) and len(result_numeric) > 0)

        input_token_length_non_numeric = "abc"
        result_non_numeric = generate_text("Test prompt", max_length=input_token_length_non_numeric)
        self.assertTrue(isinstance(result_non_numeric, str) and len(result_non_numeric) > 0)

    def test_pipeline_creation(self):
        # Test that the text generation pipeline is created successfully
        self.assertIsNotNone(self.text_generation_pipeline)

    # Integration Tests

    def test_end_to_end(self):
        # Simulate a complete interaction by providing a prompt through the input
        prompt = "End-to-End Test Prompt"
        generated_text = generate_text(prompt)
        self.assertTrue(isinstance(generated_text, str) and len(generated_text) > 0)

    def test_performance(self):
        # Test the performance of text generation with a long prompt
        prompt = "Long prompt for performance testing" * 100
        start_time = time.time()
        generate_text(prompt)
        duration = time.time() - start_time
        self.assertLess(duration, 5.0)

    def test_error_handling(self):
        # Test the behavior when the model directory is incorrect or doesn't exist
        with self.assertRaises((OSError, FileNotFoundError, transformers.modeling_utils.ConfigurationError, transformers.modeling_utils.ModelLoadConfigError)):
            model = AutoModelForCausalLM.from_pretrained("Incorrect_Directory")

if __name__ == '__main__':
    unittest.main()
