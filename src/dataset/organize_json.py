import json



root = "/ix1/hkarim/yip33/kaggle_dataset/image_net100/"

imagenet_index_path = root + "imagenet_class_index.json"
imagenet100_path = root + "imagenet100.json"
output_path = root+"imagenet100_mapped.json"

with open(imagenet_index_path, 'r') as f:
    imagenet_class_index = json.load(f)


with open(imagenet100_path, 'r') as f:
    imagenet100 = json.load(f)


imagenet100_ids = set(imagenet100.keys())


imagenet100_mapped = {}


for index, (class_id, class_name) in imagenet_class_index.items():
    if class_id in imagenet100_ids:
        imagenet100_mapped[index] = [class_id, class_name]


with open(output_path, 'w') as f:
    json.dump(imagenet100_mapped, f, indent=4)

print(f"Filtered mapping saved to {output_path}")
