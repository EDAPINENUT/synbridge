import json
import os

def load_json_data(json_file_path):
    """
    Read a JSON file and return the data array.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON file: {json_file_path}")
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        return data
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return None

def process_reaction_data(data):
    processed_data = []
    
    for i, reaction in enumerate(data):
        reaction_dict = {
            'reaction_id': i,
            'reactants': [],
            'products': [],
            'conditions': []
        }
        
        for item in reaction:
            if item['type'] == 'reactants':
                reaction_dict['reactants'].append({
                    'smiles': item['text'],
                    'relations': item['relations']
                })
            elif item['type'] == 'products':
                reaction_dict['products'].append({
                    'smiles': item['text'],
                    'relations': item['relations']
                })
            elif item['type'] == 'conditions':
                reaction_dict['conditions'].append({
                    'text': item['text'],
                    'relations': item['relations']
                })
        
        processed_data.append(reaction_dict)
    
    return processed_data

def process_json(json_file, save_dir):
    # Path to the JSON file
    
    # Load JSON data
    data = load_json_data(json_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if data is not None:
        # Process the data
        processed_data = process_reaction_data(data)
        
        # Print examples of the first few reactions
        reactions = []
        for i in range(len(processed_data)):
            reactants = []
            for reactant_info in processed_data[i]['reactants']:
                reactants.append(reactant_info['smiles'])
            products = []
            for product_info in processed_data[i]['products']:
                products.append(product_info['smiles'])
            reactions.append('.'.join(reactants) + '>>' + '.'.join(products))
        
        json.dump(
            reactions, 
            open(os.path.join(
                save_dir, 
                os.path.basename(json_file)
            ), 'w'),
            indent=2,
            ensure_ascii=False
        )

        return reactions
    
    return None
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_dir', type=str, default='/fs_mol/linhaitao/synflow_mix/preprocess/json_1012/')
    parser.add_argument('--save_dir', type=str, default='/fs_mol/linhaitao/synflow_mix/preprocess/processed_json_1012')
    args = parser.parse_args()
    json_file_dir = args.json_file_dir
    save_dir = args.save_dir
    data_list = []
    for json_file in os.listdir(json_file_dir):
        data = process_json(os.path.join(json_file_dir, json_file), save_dir)
        data_list.extend(data)
        txt_save_path = os.path.join(save_dir, "all_reactions.txt")
        with open(txt_save_path, "w", encoding="utf-8") as f:
            for rxn in data_list:
                f.write(rxn + "\n")