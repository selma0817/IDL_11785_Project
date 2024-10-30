# create conda virtual environment
# fiftyone requires python=3.9-11
conda create --name IDLSG2 python=3.11
conda activate IDLSG2

# deep learning tools
pip install torch torchsummary matplotlib

# HRNet dependencies
pip install -r requirements.txt 

