import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

class TestTextGeneration(unittest.TestCase):
    def setUp(self):
        # Set up the model and tokenizer
        self.model_directory = "Software" 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.text_generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, device=0, max_length=2000)

    def test_generate_text(self):
        prompt = "Test prompt."
        max_length = 1000  # Set the desired max_length for testing

        # Ensure the function doesn't raise any exceptions
        with self.assertLogs(level="ERROR"):
            output_text = self.text_generation_pipeline(prompt, max_length=max_length)

        # Ensure the output is not empty
        self.assertTrue(output_text)

    def test_generate_text_with_long_prompt(self):
        prompt = "This is a very long prompt. " * 50
        max_length = 2000  # Set the desired max_length for testing

        # Ensure the function doesn't raise any exceptions
        with self.assertLogs(level="ERROR"):
            output_text = self.text_generation_pipeline(prompt, max_length=max_length)

        # Ensure the output is not empty
        self.assertTrue(output_text)

if __name__ == "__main__":
    unittest.main()