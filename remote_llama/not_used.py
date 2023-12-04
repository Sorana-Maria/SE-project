import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import time

# Set the model name and directory where the model is saved
model_directory = "Software"  # Adjust this if the model is saved in a different directory

# Check if multiple GPUs are available, otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    torch_dtype=torch.float16,  # Use float16 to save GPU memory
    low_cpu_mem_usage=True,
    device_map='auto'  # Automatically maps the model layers across all available GPUs
)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Function to generate text based on input prompts
def generate_text(prompt, max_length=2000):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Move inputs to GPU
    outputs = model.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0].cpu().numpy())  # Move output tensor back to CPU for decoding

# Interactive prompt for user input
#input_str = input("Enter your prompt (or type 'exit' to quit): ")
input_str= input("""enter input:  """)
print(f"input length is {len(input_str)} characters")
input_token_length = input("Enter max token length for generation (default 2000): ")
input_token_length = int(input_token_length) if input_token_length.isdigit() else 2000

time_start = time.time()

output_str = generate_text(input_str, max_length=input_token_length)

print(output_str)

print(f"input length is {len(output_str)} characters")
print("Time taken: ", time.time() - time_start)

print("Time taken: ", -timeStart + time.time())
generated_text=output_str

# Initialize the tokenizer and model for T5
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

# Translate the generated text
# Specify the target language for translation
target_language = "fr"  # Replace with your desired language code (e.g., "fr" for French)

# Encode the text to be translated by T5
translation_input = t5_tokenizer(f"translate English to {target_language}: {generated_text}", return_tensors="pt").input_ids.to(device)

# Generate the translation using the T5 model
translation_output = t5_model.generate(translation_input, max_length=len(output_str)*1.1)
translated_text = t5_tokenizer.decode(translation_output[0], skip_special_tokens=True)

print(f"Translated text: {translated_text}")