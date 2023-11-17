import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if GPU is available, otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timeStart = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,  # Use float16 to save GPU memory
    low_cpu_mem_usage=True,
    device_map = 'auto',
)



print("Load model time: ", -timeStart + time.time())

input_str = input('Enter: ')
input_token_length = input('Enter length: ')


timeStart = time.time()

inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)  # Move inputs to GPU

outputs = model.generate(
    inputs,
    max_length=int(input_token_length),
)

output_str = tokenizer.decode(outputs[0].cpu().numpy())  # Move output tensor back to CPU for decoding

print(output_str)

print("Time taken: ", -timeStart + time.time())
