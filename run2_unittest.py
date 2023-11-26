import unittest

class TestTextGeneration(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test case
        self.model_directory = "Software"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.text_generation_pipeline = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, device=0, max_length=2000)

    def tearDown(self):
        # This method will be called after each test case
        pass

    def test_generate_text(self):
        # Test the generate_text function with a simple prompt
        prompt = "Test prompt."
        generated_text = generate_text(prompt)
        self.assertTrue(isinstance(generated_text, str))

    def test_max_token_length(self):
        # Test if the max token length is working as expected
        prompt = "Test prompt."
        max_length = 1000
        generated_text = generate_text(prompt, max_length=max_length)
        tokens = tokenizer(generated_text)["input_ids"]
        self.assertTrue(len(tokens) <= max_length)

if __name__ == '__main__':
    unittest.main()