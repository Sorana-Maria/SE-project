import os
import time
#import openai
import random
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import pipeline
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ccdv/arxiv-summarization")

# Extract the training and validation datasets
train_dataset = dataset['train']
valid_dataset = dataset['validation']
num_shards = 1000  # 100% / 5% = 20
train_dataset = train_dataset.shard(num_shards, index=0)  # Take the first shard (5% of the dataset)
valid_dataset = valid_dataset.shard(num_shards, index=0)  # Take the first shard (5% of the dataset)

import pandas as pd
"""
# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples. Here are the first few:')

df.head()
"""

# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-hf"

# The instruction dataset to use
dataset_name = "/home/u590531/llm/data.jsonl"

# Fine-tuned model name
new_model = "Llama-2-7b-fintuned"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_hf"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 2048

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
#device_map = "auto"
device_map='auto'

### loading dataset
# Load datasets
#train_dataset = load_dataset('json', data_files='/home/u590531/llm/train.jsonl', split="train")
#valid_dataset = load_dataset('json', data_files='/home/u590531/llm/test.jsonl', split="train")
#print(valid_dataset)
# Preprocess datasets

# Map the datasets
train_dataset_mapped = train_dataset.map(
    lambda examples: {'text': [article + ' ' + abstract for article, abstract in zip(examples['article'], examples['abstract'])]},
    batched=True
)

valid_dataset_mapped = valid_dataset.map(
    lambda examples: {'text': [article + ' ' + abstract for article, abstract in zip(examples['article'], examples['abstract'])]},
    batched=True
)

#train_dataset_mapped = train_dataset
#valid_dataset_mapped = valid_dataset
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    #report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
# Save the model and tokenizer
save_directory = "/home/u590531/llm/Software/"  # Specify the path where you want to save the model
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Cell 4: Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = """1 Introduction
A recent report by the World Wide Fund for Nature (WWF) confirms that biodiversity and ecosystems
are deteriorating worldwide [1]. Population sizes of mammals, birds, amphibians, reptiles and fish
have decreased by an average of 68 percent between 1970 and 2016 across the world. This decrease in
biodiversity has several causes, such as habitat loss due to pollution, species overexploitation or
climate change. Biodiversity is important since it is a key indicator of overall healthy ecosystems
which in their turn have important social and economic consequences for humans. In particular,
biodiversity and ecosystems influence our water quality, air quality and climate, they secure our food
production and impact the spread of infectious diseases originating from animals [1, 5].
Machine learning (ML) can help to more efficiently measure and monitor the well-being of ecosystems
and the success of biodiversity conservation efforts [14, 16, 35, 25]. As an example, this paper
proposes a method for automatic classification of camera trap images, a type of motion triggered
cameras used in biological studies to estimate animal population density and activity patterns [26,
∗
Joint first author.
†Corresponding author.
35th Conference on Neural Information Processing Systems (NeurIPS 2021), Sydney, Australia.
7, 27, 29, 32, 34]. Since manually labeling large numbers of camera trap images is time consuming
and costly [17], ML could be used to automatically detect animals and the species to which they
belong in images. This work uses Convolutional Neural Networks [18, 19] to classify camera trap
images. Training a CNN on a dataset of camera trap images is challenging, because camera trap
images often only depict a part of an animal, because of high intra-class variation due to differences in
backgrounds, and because the class-distribution of camera trap datasets is typically highly imbalanced.
This imbalance is inherent to camera trap datasets since it reflects the imbalance of ecosystems [33],
and it results in biased classifiers that perform very well for a few majority classes but poorly for many
minority classes. Classifiers that perform well on all classes would be of more value to ecologists,
and moreover, rare or less observed animal species might even be of special interest to research.
Therefore, solutions are needed to mitigate this imbalance when classifying camera trap images.
To this end, we use a two-phase training method [20] to mitigate class imbalance, for the first time to
the best of our knowledge on camera trap images. In experiments we compare it to different data-level
class imbalance mitigation techniques, and show that it improves performance on minority classes,
with limited loss in performance for other classes, resulting in an increase in macro F1-score.
2 Related work
Pioneering studies that automatically classified camera trap images relied on manual feature extraction
and smaller datasets [6, 31, 39, 4]. Better and more scalable results were later achieved with deep
CNNs and larger datasets [8, 24, 32, 38, 28]. Generally, models trained by these scholars achieve
accuracies well above 90%, but the models are biased towards majority classes, which severely affects
their class-specific performance. Especially the performance for rare species is poor. Scholars dealt
with this challenge by removing the rare classes from the dataset [8, 38], with confidence thresholding
and letting experts review the uncertain classifications [38], with weighted losses, oversampling and
emphasis sampling [24] or by using a combination of additional image augmentations for rare classes
and novel sampling techniques [28]. Although [24] managed to greatly increase the accuracy for a
few rare classes using oversampling, none of the aforementioned techniques systematically improved
accuracy for most of the rare species. It can thus be concluded that dealing with class-imbalance in
the context of camera trap image classification is still an unsolved issue.
Two categories of methods for mitigation of class imbalance in deep learning exist: data-level and
algorithm-level techniques [2, 15]. The former refers to techniques that alter the class-distribution of
the data, such as random minority oversampling (ROS) and random majority undersampling (RUS),
which respectively randomly duplicate or randomly remove samples to obtain a more balanced dataset.
More advanced techniques can also be used to synthesize new samples [3, 9, 12, 36, 37, 21], but
these are computationally expensive, and they require a large number of images per class and images
within a class that are sufficiently similar. Algorithm-level techniques are techniques that work on
the model itself, such as loss functions or thresholding [22, 23, 2, 15, 11, 2]. Two-phase training,
a hybrid technique, was recently introduced and shown to obtain good results for training a CNN
classifier on a highly imbalanced dataset of images of plankton [20], and it was later used by others
for image segmentation and classification [10, 2]. Because of these promising results and the broad
applicability of 2-phase training, we test 2-phase training for camera trap images.
3 Two-phase training
Two-phase training consists of the following steps [20]. Dorig is the original, imbalanced dataset.
Figure 3 in the appendix shows an overview of two-phrase training.
1. Phase 1: a CNN fθ is trained on a more balanced dataset Dbal, obtained by any sampling
method such as ROS, RUS or a combination thereof.
2. Phase 2: the convolutional weights3 of fθ are frozen, and the network is trained further on
the full imbalanced dataset Dorg.
The 1st phase trains the convolutional layers with (more) equal importance allocated to minority
classes, so they learn to extract relevant features for these classes as well. In the 2nd phase the
classification layers learn to model the class imbalance present in the dataset.
3
I.e. all weights except the weights of the fully connected layers that project the CNN features to the classes.
2
Model Phase 1: Accuracy Phase 2: Accuracy Phase 1: F1 Phase 2: F1
Dorig: Baseline 0.8527 / 0.3944 /
D
1
bal: ROS 0.8326 0.8528 0.3843 0.4012
D
2
bal: RUS 0.8012 0.8491 0.3681 0.4147
D
3
bal: ROS&RUS(15K) 0.8346 0.8454 0.4179 0.4094
D
4
bal: ROS&RUS(5K) 0.7335 0.8066 0.3620 0.4001
Table 1: Model Comparison - Top-1 accuracy and Macro F1-score.
4 Dataset & Experiments
We used the 9th season of the publicly available Snapshot Serengeti (SS) dataset, which is generated
by a grid of 225 cameras spread over the Serengeti National Park in Tanzania [30]. The images were
labeled by citizen scientists on the Zooniverse platform. After filtering some images, the full dataset
Dorig contains 194k images belonging to 52 classes. The class-distribution of this dataset is depicted
in fig. 4 in the appendix, and is highly imbalanced, with the three majority classes accounting for
just under 75% of the data. We used this smaller subset of the full SS dataset for computational
tractability, and to ensure insights remain valid for ecologists with access to smaller datasets.
Appendix A.2 lists the hyperparameters4
. First we trained the baseline CNN on the full dataset Dorig.
Next, we trained 4 models with different instantiations of Dbal for phase 1 of two-phase training.
1. D1
bal: ROS (oversampling) classes with less than 5k images until 5k, see appendix fig. 5.
2. D2
bal: RUS (undersampling) classes with more than 15k images until 15k.
3. D3
bal: ROS classes with less than 5k images until 5k as in 1., and RUS classes with more
than 15k images until 15k as in 2. Shown in fig. 6 in the appendix.
4. D4
bal: ROS classes with less than 5k images until 5k as in 1., and RUS classes with more
than 5k images until 5k.
We used a lower sample ratio for classes with very few images to avoid overfitting (appendix A.3).
As evaluation metrics we used not only top-1 accuracy but also precision, recall and F1-score, since
these metrics are more informative to class-imbalance. We report their values macro-averaged over
classes as well as the class specific values (in appendix tables 4-6). The results of the models after
phase 1 correspond to the results that we would obtain by only using ROS, RUS or a combination of
both (and no two-phase training). These results will thus serve as a baseline.
5 Results
Accuracy and Macro F1. Table 1 shows the accuracy and F1-score of the models after the 1st
and the 2nd phase5
. Training on more balanced datasets reduces accuracy in phase 1 for all models
compared to the baseline which was trained on the imbalanced dataset Dorig. However, further
training the classification layers in phase 2 on the full dataset increases accuracy back to more or less
the baseline level for all models (except ROS&RUS(5K)), meaning that two-phase training lost little
to no accuracy. The phase 2 mean accuracy is substantially higher than the phase 1 mean accuracy.
The F1-scores of most models also drop in phase 1. Interestingly, phase 2 raises the F1-score of
most models again, and all models obtain an F1-score after phase 2 that is higher than the baseline:
3.0% on average. The RUS model obtains the highest F1-score after phase 2: an increase of 5.1%
compared to the baseline, while the ROS&RUS(15K) model obtain the highest F1-score overall6
.
Most two-phase trained models outperform their counterparts which were only trained on more
balanced datasets. As for the accuracy, the mean F1-score in phase 2 is substantially higher than the
mean F1-score in phase 1: 6.1%.
These observations lead us to conclude that 1) two-phase training outperforms using only sampling
techniques across most sampling regimes, and 2) two-phase training can increase the F1-score without
4Our code is publicly available: https://github.com/FarjadMalik/aigoeswild.
5Appendix A.4 contains more results and in-depth discussion.
6We consider the F1-score of ROS&RUS(15K) after phase 1 an anomaly which needs further analysis.
3
(a) (b)
Figure 1: Relative difference in F1-score per species of (a) the two-phase RUS model vs. the baseline,
and (b) phase 2 vs. phase 1 of the RUS-model. The appendix contains larger versions: figs. 8, 10.
Species are sorted in descending order according to their occurrence frequency.
substantial loss in accuracy, meaning it improves minority class predictions with very limited loss in
majority class performance. These findings are in line with the results of [20], though they report
greater increases in F1-scores than us, possibly due to an even more imbalanced dataset. They also
find RUS to work best for creating Dbal for phase 1. The F1-scores are substantially lower than the
accuracies (idem for precision and recall, appendix tables 2-3). This is because the class-specific
values for these metrics are high for the majority classes, but extremely low for many minority classes,
confirming that the imbalanced data creates a bias towards the majority classes.
Class-specific performance. Class-specific F1-scores increase with two-phase training for the
majority of (minority) classes. Two-phase training with RUS leads to the greatest average increase of
F1-score per class: 3% (ignoring the classes for which the F1-score remained 0.0%). This increase is
most notable for minority classes. RUS performing best is remarkable, since we trained the RUS
model in phase 1 with only 85k images, compared to 131k–231k for the other models. Fig. 1a shows
the changes in F1-score due to two-phase training with RUS.
6 Conclusion
We explored the use of two-phase training to mitigate the class imbalance issue for camera trap
image classification with CNNs. We conclude that 1) two-phase training outperforms using only
sampling techniques across most sampling regimes, and 2) two-phase training improves minority
class predictions with very limited loss in majority class performance, compared to training on
the imbalanced dataset only. In the future we would like to rerun our experiments with different
random seeds to obtain more statistically convincing results, compare two-phase training to other
algorithm-level imbalance mitigation techniques, and test it on varying dataset sizes and levels of
class imbalance.""" # replace the command here with something relevant to your task
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=10000)
result = pipe(prompt)
print(result[0]['generated_text'])

while True:
    input_str = input('Enter: ')
    input_token_length = input('Enter length: ')

    if input_str.lower() == 'exit':
        break

    timeStart = time.time()

    inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)  # Move inputs to GPU

    outputs = model.generate(
        inputs,
        max_length=int(input_token_length),
    )

    output_str = tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True)  # Move output tensor back to CPU for decoding and skip special tokens during decoding

    print(output_str)

    print("Time taken: ", time.time() - timeStart)


while True:
    input_str = input('Enter: ')
    input_token_length = input('Enter length: ')

    if input_str.lower() == 'exit':
        break

    timeStart = time.time()

    # Using the pipeline for text-generation
    result = pipe(input_str, max_length=int(input_token_length))

    output_str = result[0]['generated_text']

    print(output_str)

    print("Time taken: ", time.time() - timeStart)