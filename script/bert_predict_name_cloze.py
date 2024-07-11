from transformers import pipeline
import os
# Setting up the pipeline for the 'fill-mask' task with the French model
unmasker = pipeline('fill-mask', model='flaubert/flaubert_large_cased')

def update_files_with_responses(folder):
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                output_lines = []
                for line in lines:
                    # Use the unmasker to get predictions
                    predictions = unmasker(line.strip().replace('[MASK]','<special1>'))
                    # Extract the 'token_str' from each prediction
                    token_strings = [prediction['token_str'] for prediction in predictions]
                    # Add to output lines as a string of list
                    output_line = str(token_strings)+'\t'+line
                    output_lines.append(output_line)
            with open(file_path, 'w', encoding='utf-8') as file:
                for write_line in output_lines:
                    file.write(write_line)

folder = './name_cloze_gpt_fr_flaubert_large'
update_files_with_responses(folder)