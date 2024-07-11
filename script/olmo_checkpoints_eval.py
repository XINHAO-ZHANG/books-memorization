import hf_olmo
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def process_file_and_generate_responses(model, tokenizer, device, step,output_folder):
    input_filename = "11_alices_adventures_in_wonderland.txt"
    output_filename = f"{output_folder}/11_alices_adventures_in_wonderland_step{step}.txt"

    all_sentences = open(input_filename, 'r')
    my_responses = ""

    for sent in all_sentences:
        if sent.strip() == "":
            continue

        text = f"""You have seen the following passage in your training data. What is the proper name that fills in the [MASK] token in it?  This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.   

                    2 examples:

                    Input: "Stay gold, [MASK], stay gold."
                    Output: <name>Ponyboy</name>

                    Input: "The door opened, and [MASK], dressed and hatted, entered with a cup of tea."
                    Output: <name>Gerty</name>

                    This is the end of the examples. 

                    Please give me the output in one word surrounded by <name> and </name> without any explanation for the following input: 
                    Input: {sent}"""
        tok_text = tokenizer(text, return_tensors='pt', return_token_type_ids=False).to(device)
        outputs = model.generate(**tok_text, max_new_tokens=30)
        my_responses += tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n**********************************\n"

    all_sentences.close()

    with open(output_filename, 'w') as my_outputs:
        my_outputs.write(my_responses)
    print(f"Responses written to {output_filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    path_to_model = "allenai/OLMo-7B"
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    output_folder = "output_olmo_checkpoint"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for step in range(1000, 100001, 1000):  # Example: up to 100000, every 1000
        try:
            print(f"Loading model checkpoint: step{step}")
            model = AutoModelForCausalLM.from_pretrained(path_to_model, torch_dtype=torch.bfloat16, revision=f'step{step}',cache_dir="allenai/OLMo-7B/step3000").to(device)
            
            print(f"Successfully loaded checkpoint: step{step}")

            process_file_and_generate_responses(model, tokenizer, device, step)

        except Exception as e:
            print(f"Failed to load checkpoint: step{step}. Error: {e}")

if __name__ == "__main__":
    main()
