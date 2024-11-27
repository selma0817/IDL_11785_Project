

## rename folder from classname to idx for kaggle image_net100 dataset

1. run `organize_json.py first` to get `imagenet100_mapped.json`
this `imagenet100_mapped.json` is a mapping that maps each folder name in the original data from kaggle, such as `n01968897` to its corresponding idx like `0`. You need to change the root to 
where you download the kaggle imagenet100 data
2. run `unpack.py` to rename the folder of kaggle imagenet100 data to idx so we can use ray's dataloader

