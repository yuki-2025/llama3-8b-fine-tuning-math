# Fine-tuning Llama3 8b for Math

fine-tuning Llama3 8b to generate JSON formats for arithmetic questions and process the output to perform calculations. 
This method incorporates the latest fine-tuning techniques such as Qlora, Unsloth, and PEFT. 
It enables faster training speeds and requires fewer computational resources.

PS: You need a T4 (16GB) GPU to run the code.

Colab Live code: https://drive.google.com/file/d/1NsSS1_M3pNAbkiBnPB3k5JKIkEQg3XNX/view?usp=sharing

## Setup 
1. Download all the files in this repo.
2. Run ```Load_model.py``` to load library and Llama3 8b.
3. Run ```Prepare_data.py``` to load ```function_call.jsonl``` dataset and prepare dataset.
4. Run ```Fine_tuning.py ```
5. Run ```Inference_n_save.py  ``` to test the fine-tuned models and save the model.
 

## Preview 

### Before Fine-tuning:  </br>
**Test1:**</br>
<img src="https://github.com/yuki-2025/llama3-8b-fine-tuning-math/assets/159591455/084a454d-5310-42d0-a5a5-8b53bcad7b79" alt="Untitled design" width="800" height="500"></br>
**Test2:**</br>
<img src="https://github.com/yuki-2025/llama3-8b-fine-tuning-math/assets/159591455/08661ac4-d97f-49fd-a9ed-6c5988f7e239" alt="Untitled design" width="600" height="300"></br>
**Test3:**</br>
<img src="https://github.com/yuki-2025/llama3-8b-fine-tuning-math/assets/159591455/4bb4a4dd-4bb8-43be-8007-0675c7c44ba3" alt="Untitled design" width="800" height="300"></br>

### After Fine-tuning:</br>
<img src="https://github.com/yuki-2025/llama3-8b-fine-tuning-math/assets/159591455/e331d367-d441-4f14-94eb-62bfe1e863e0" alt="Untitled design" width="600" height="400"></br></br>

This is part of my research study in The University of Chicago. The data is come from:
https://github.com/rohanbalkondekar/finetune_llama2
