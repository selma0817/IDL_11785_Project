import json
import os

def rename_folders(base_dir, train='train_data', val='val_data', class_json='imagenet100_mapped.json'):
    train_dir = os.path.join(base_dir, train)
    val_dir = os.path.join(base_dir, val)
    json_path = os.path.join(base_dir, class_json)

    with open(json_path, 'r') as json_file:
        class_mapping = json.load(json_file)

    synset_to_idx = {v[0]: k for k, v in class_mapping.items()}

    def rename_in_place(data_dir):
        for synset in os.listdir(data_dir):
            synset_path = os.path.join(data_dir, synset)

            if not os.path.isdir(synset_path):
                continue

            # Get the index for the synset ID
            idx = synset_to_idx.get(synset, None)
            if idx is not None:
                new_folder_name = os.path.join(data_dir, idx)

                # Rename the folder
                if not os.path.exists(new_folder_name):
                    os.rename(synset_path, new_folder_name)
                    print(f"Renamed {synset} -> {idx}")
                else:
                    print(f"Warning: {new_folder_name} already exists. Skipping {synset}.")
            else:
                print(f"Warning: {synset} not found in JSON mapping. Skipping.")

    # Rename folders in train and val directories
    print("Renaming folders in the train directory...")
    rename_in_place(train_dir)

    print("Renaming folders in the val directory...")
    rename_in_place(val_dir)

    print("Renaming complete.")

if __name__ == '__main__':
    base_dir = '/home/ray/proj/IDL_11785_Project/data/imagenet100'
    rename_folders(base_dir)
