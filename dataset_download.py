import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    'coco-2017',
    split = ['train', 'test', 'validation'],
    
)