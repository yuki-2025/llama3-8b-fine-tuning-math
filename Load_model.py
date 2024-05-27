%%capture 
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
  
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",  
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# prompt = input("Ask Question: ")
formatted_prompt = f"Reply with JSON for the following question: what is the answer of Sine of 60 degrees"

# Generating the input for the model
inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
gen_tokens = model.generate(**inputs,  max_new_tokens=128)

generated_text = tokenizer.batch_decode(gen_tokens)[0]
print(generated_text)
