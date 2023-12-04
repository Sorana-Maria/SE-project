import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Check if there are multiple GPUs available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timeStart = time.time()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")  # <-- assuming the 7b version is correct

# Load the model with specific parameters
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

print("Load model time: ", time.time() - timeStart)

# Infinite loop for continuous inference
instert_str= """" #####
summarize the text in between -- and -- ,text: -- British logistics supported the operations of Field Marshal Sir Bernard Montgomery's Anglo-Canadian 21st Army Group in the Western Allied invasion of Germany from 8 January 1945 until the end of the Second World War in Europe on 8 May 1945. To conserve scarce manpower, the British and Canadian forces employed mechanisation and materiel for maximum effect in combat operations. This involved prodigious use of ammunition, fuel and equipment, which in turn demanded a first-class military logistics system. By this time, the British Army was highly experienced, professional and proficient.

Originally scheduled to start at the beginning of January 1945, when the ground would have been frozen, Operation Veritable, the 21st Army Group's advance to the Rhine, was delayed for five weeks by the German Ardennes Offensive. It was therefore conducted over muddy and sometimes flooded ground, and roads were sometimes impassable even to four-wheel-drive vehicles. The offensive was supported by 600 field and 300 medium guns. Over 2.5 million rounds of 25-pounder ammunition were made available. The army roadheads were mainly supplied by rail. Fuels were brought by tankers and the Operation Pluto pipeline from the UK, and delivered by barge and pipeline to the army roadheads. Special arrangements were made to supply the Royal Air Force's Fog Investigation and Dispersal Operation, which consumed 410,000 litres (90,000 imp gal) a night, and the Gloster Meteor jet fighters, which consumed 14,000 litres (3,000 imp gal) of kerosene each day. Montgomery's armies were reinforced by the redeployment of three divisions from Italy under Operation Goldflake.
eeks by the German Ardennes Offensive. It was therefore conducted over muddy and sometimes flooded ground, and roads were sometimes impassable even to four-wheel-drive vehicles. The offensive was supported by 600 field and 300 medium guns. Over 2.5 million rounds of 25-pounder ammunition were made available. The army roadheads were mainly supplied by rail. Fuels were brought by tankers and the Operation Pluto pipeline from the UK, and delivered by barge and pipeline to the army roadheads. Special arrangements were made to supply the Royal Air Force's Fog Investigatio

The next major operation was Operation Plunder—the assault crossing of the Rhine on 23 March. For this the British Second Army and the US Ninth Army deployed 2,144 field and medium guns, augmented by 3,337 anti-aircraft guns and anti-tank guns. A large force of engineer units was assembled for the operation: 37,000 British and Canadian engineers and pioneers, and 22,000 American engineers. Every available amphibious craft was collected, and they were joined by a Royal Navy contingent of 36 LCMs and 36 LCVPs that were transported by road across Holland and Belgium to participate. Operation Plunder included an airborne operation, Operation Varsity, in which two airborne divisions were landed with a day's supply of food, fuel and petrol. Engineers soon had bridges in operation over the Rhine that were later superseded by more permanent road and rail bridges.

During the first three weeks of April 1945, the 21st Army Group advanced about 320 kilometres (200 mi) across northern Germany to reach the Elbe on 19 April and then the Baltic Sea. Until the railway bridges could be brought into operation, maintenance depended entirely on road transport. The 21st Army Group allocated further road transport capacity to the armies by shifting vehicles from the rear areas and immobilising units that were not immediately needed. The corps sometimes had to send their transport back to the army roadheads to assist when major operations were required. The high use of road transport meant that the Second Army burned 7,600 tonnes (7,500 long tons) of petrol a day, but pipelines were laid across the Rhine at Emmerich and were in operation by the end of April. On 4 May, Montgomery took the surrender of the German forces in front of the 21st Army Group.   --
 start and end summarization with: ###### """
input_str = instert_str
print("Input string length:")
print(len(input_str))
input_token_length = len(input_str)/3

# Tokenize the input string and move the tokens to the GPU
inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)

# Generate the output using the model
outputs = model.generate(
    inputs,
    max_length=int(input_token_length),
)

# Decode the output tokens to a string and print it
output_str = tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True)
print(output_str)

print("Time taken: ", time.time() - timeStart)
