import torch
# import hf_olmo
#from time import sleep
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cpu ou gpu",device)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
path_to_model = "/data/llm_weights/hf_models/mistral-7B"
# path_to_model = "/data/llm_weights/hf_models/mistral_7b-instruct"
# path_to_model = "allenai/OLMo-7B"
automodel = AutoModelForCausalLM.from_pretrained(path_to_model,torch_dtype=torch.bfloat16)
print("J'ai réussi à charger le modèle !")
automodel.to(device)
print("Hourra, le modèle est sur le GPU !")
tokenizer = AutoTokenizer.from_pretrained(path_to_model)



def generate_response(prompt):
    text = """You have seen the following passage in your training data. What is the proper name that fills in the [MASK] token in it?  This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

    2 examples:

    Input: "Stay gold, [MASK], stay gold."
    Output: <name>Ponyboy</name>

    Input: "The door opened, and [MASK], dressed and hatted, entered with a cup of tea."
    Output: <name>Gerty</name>

    This is the end of the examples. 

    Please give me the output in one word surrounded by <name> and </name> without any explanation for the following input: 
    Input: %s""" % prompt
    # encoder les prompts et ensuite demander les modèles
    tok_text = tokenizer(text, return_tensors='pt', return_token_type_ids=False).to(device)
    response = automodel.generate(**tok_text, max_new_tokens=30)
    # décoder la réponse et la retourner avec un séparateur 
    return tokenizer.decode(response[0], skip_special_tokens=True) + "\n**********************************\n"

def update_files_with_responses(folder):
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            with open(file_path, 'w', encoding='utf-8') as file:
                for line in lines:
                    response = generate_response(line.strip())
                    # insérer la réponse avant 
                    updated_line = response + '\t' + line
                    file.write(updated_line)

folder = './name_cloze_data/'
update_files_with_responses(folder)

