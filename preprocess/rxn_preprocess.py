import argparse
from rxnmapper import RXNMapper
import os
rxn_mapper = RXNMapper()
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--input_file', type=str, default='/fs_mol/linhaitao/synflow_mix/preprocess/processed_json_1012/all_reactions.txt')
parser.add_argument('--output_file', type=str, default='/fs_mol/linhaitao/synflow_mix/preprocess/processed_json_1012/all_reactions_rxnmapper.csv')
parser.add_argument('--confidence_min', type=float, default=0.5)
args = parser.parse_args()
with open(args.input_file, 'r', encoding='utf-8') as f:
    all_reactions = [line.strip() for line in f if line.strip()]
batch_size = args.batch_size
confidence_min = args.confidence_min
all_results = []
error_reactions = []
filtered_reactions = []
for i in range(0, len(all_reactions), batch_size):
    try:
        batch_reactions = all_reactions[i:i+batch_size]
        results = rxn_mapper.get_attention_guided_atom_maps(batch_reactions)
        all_results.extend(results)
        print(f"Processed {i+batch_size} reactions")        
    except:
        print(f"Error processing {i+batch_size} reactions, process one by one...")
        batch_reactions = all_reactions[i:i+batch_size]
        for j, reaction in enumerate(batch_reactions):
            try:
                results = rxn_mapper.get_attention_guided_atom_maps([reaction])
                all_results.extend(results)
            except:
                print(f"Error processing {i+j} reactions, which is {reaction}")
                error_reactions.append(reaction)

filter_reaction_index = []           
for i, result in enumerate(all_results):
    if result['confidence'] >= confidence_min:
        filter_reaction_index.append(i)
        filtered_reactions.append(result['mapped_rxn'])

import csv
with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

with open(os.path.join(os.path.dirname(args.output_file), f'error_reactions.txt'), 'w', encoding='utf-8') as f:
    for reaction in error_reactions:
        f.write(reaction + "\n")

with open(os.path.join(os.path.dirname(args.output_file), f'filtered_reactions_{confidence_min}.txt'), 'w', encoding='utf-8') as f:
    for reaction in filtered_reactions:
        f.write(reaction + "\n")

with open(os.path.join(os.path.dirname(args.output_file), f'filtered_reactions_index_{confidence_min}.txt'), 'w', encoding='utf-8') as f:
    for index in filter_reaction_index:
        f.write(str(index) + "\n")
    


