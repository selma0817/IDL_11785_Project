# create conda virtual environment
# fiftyone requires python=3.9-11
conda create --name IDLSG2 python=3.11
conda activate IDLSG2

# deep learning tools
pip install torch torchsummary matplotlib seaborn pandas pyyaml

# COCO
pip install pycocotools

# download from google drive
pip install gdown

# make models and data directories
mkdir metrics metrics/plots
mkdir models 
mkdir data data/images
