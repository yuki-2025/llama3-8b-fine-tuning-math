import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value, Sequence, DatasetInfo
import pyarrow as pa
import json

# Function to read and parse JSONL file into a list of dictionaries
def load_jsonl_to_dataset(jsonl_path):
    with open(jsonl_path, 'r') as file:
        jsonl_data = [json.loads(line) for line in file]
        # Convert outputs from string in markdown format to a plain dictionary format
        for entry in jsonl_data:
            entry['output'] = json.loads(entry['output'].split('```json\n')[1].split('\n```')[0])
    return jsonl_data

# Load your JSONL data
jsonl_data = load_jsonl_to_dataset('/content/function_call.jsonl')  # Provide the correct path to your JSONL file
jsonl_dataset = Dataset.from_pandas(pd.DataFrame(jsonl_data))
# Function to ensure output is in string format
def ensure_string_output(entry):
    if isinstance(entry['output'], dict):
        return json.dumps(entry['output'])  # Serialize dictionary to string
    return str(entry['output'])  # Convert any non-string data to string

# Adjust jsonl_data to ensure output is a string
for entry in jsonl_data:
    entry['output'] = ensure_string_output(entry)

# # Combine the datasets
# Prepare data for creating a Dataset object
data_dict = {
    "instruction": [d['instruction'] for d in jsonl_data],
    "input":  [""] * len(jsonl_data),  # Assuming no inputs in JSONL data
    "output":  [d['output'] for d in jsonl_data]
}

# Create a schema for the dataset
features = Features({
    'instruction': Value('string'),
    'input': Value('string'),  # assuming 'input' is always a string; adjust as necessary
    'output':  Value('string'),
})

# Create a DataFrame and then convert to Dataset to handle complex nested structures
df = pd.DataFrame(data_dict)
combined_dataset = Dataset.from_pandas(df, features=features)
display(df)
display(combined_dataset)


# Map the formatting function
dataset = combined_dataset.map(formatting_prompts_func, batched=True)

# Display an example or save the dataset
print(dataset['text'][0])  # Show the first processed entry
