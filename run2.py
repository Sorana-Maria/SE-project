import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import time

# Set the model name and directory where the model is saved
model_directory = "Software"  # Adjust this if the model is saved in a different directory

# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ensure that the model uses the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create a pipeline for text generation
text_generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0, max_length=2000)

# Function to generate text based on input prompts
def generate_text(prompt, max_length=2000):
    result = text_generation_pipeline(prompt, max_length=max_length)
    return result[0]['generated_text']

# Interactive prompt for user input

#input_str = input("Enter your prompt (or type 'exit' to quit): ")

input_str = input("Enter your prompt (or type 'exit' to quit): ")
input_token_length = input("Enter max token length for generation (default 2000): ")
input_token_length = int(input_token_length) if input_token_length.isdigit() else 2000

time_start = time.time()

output_str = generate_text(input_str, max_length=input_token_length)

print(output_str)
print("Time taken: ", time.time() - time_start)
