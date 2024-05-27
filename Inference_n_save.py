import re, json, math

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Yuki20/Alpaca_8b_unsloth_json",  
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

def evaluate_json(json_data):
    function_name = json_data.get("function_name")
    parameter_1 = float(json_data.get("parameter_1", 0))
    parameter_2 = float(json_data.get("parameter_2", 0))

    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
        "square_root": lambda x: math.sqrt(x),
        "cube_root": lambda x: x ** (1/3),
        "sin": lambda x: math.sin(math.radians(x)),
        "cos": lambda x: math.cos(math.radians(x)),
        "tan": lambda x: math.tan(math.radians(x)),
        "log_base_2": lambda x: math.log2(x),
        "ln": lambda x: math.log(x),
        "power": lambda x, y: x ** y,
        "log_base_10": lambda x: math.log10(x)
    }

    if function_name in operations:
        if 'parameter_2' in json_data:
            return operations[function_name](parameter_1, parameter_2)
        else:
            return operations[function_name](parameter_1)
    return None 

while True:
    prompt = input("Ask Question: ")
    formatted_prompt = f"Response with json for the following problem: {prompt}"

    # Generating the input for the model
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    gen_tokens = model.generate(**inputs,  max_new_tokens=128)

    generated_text = tokenizer.batch_decode(gen_tokens)[0]

    print("\nGenerated JSON:\n" + "-"*20)
    #print(generated_text)

    # Extract JSON from the generated text
    # Search for the pattern in the text
    match = re.search(r'\{.*?\}', generated_text) 
    # Check if a match was found and print it
    if match:
        json_string = match.group()
        print(json_string)
        json_data = json.loads(json_string) 
        print("The Answer: ", evaluate_json(json_data))
    else:
        print("No JSON format found in the text.")
 
