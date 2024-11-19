import yaml
from yacs.config import CfgNode as CN

# Define the default configuration using CfgNode
def get_default_config():
    cfg = CN()
    cfg.model = 'swinv2'
    cfg.pretrained = ''

    cfg.training = CN()
    cfg.training.batch_size = 64
    cfg.training.epochs = 30
    cfg.training.learning_rate = 0.001
    cfg.training.optimizer = 'adam'

    return cfg

# Load YAML configuration and merge with defaults
def load_config(cfg, yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_cfg = yaml.safe_load(file)
    cfg.merge_from_other_cfg(CN(yaml_cfg))

# Initialize and load the configuration
default_cfg = get_default_config()
(default_cfg, 'config.yaml')
