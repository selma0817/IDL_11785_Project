## Downloading imagnet100
1. Download imagenet100 dataset from kaggle. You should see train.X1, train.X2, train.X3, train.X4, val.X1 after unzipping the file. (unzipp them in `IDL_11785_Project/data/imagenet100`)
2. Navigate to `src`. Copy imagenet100_mapped.json to `IDL_11785_Project/data`.
3. Run unpack.py

## Note about using imagenet100 
There are 100 classes in imagenet100. While the folders are numbered from 0-150 (some class numbers are omitted), the dataloader will relabel the class numbers to exactly 0-99. (Essentially, the dataloader doesn't care about the names of the folders (they are 0-150 but could very well be just strings). It will relabel the folders based on their order.) Don't make num_classes=150, 100 is enough.