# create conda virtual environment
# fiftyone requires python=3.9-11
conda create --name IDLSG2 python=3.11
conda activate IDLSG2

# deep learning tools
pip install torch torchsummary matplotlib seaborn pandas
# calflops for calculating flops
pip install calflops 

# isntall conda api
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi/PythonAPI
# Install into global site-packages
# make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
