# navigate to root directory
cd models
mkdir hrnet hrnet/configs hrnet/paths
cd hrnet/paths
gdown --folder https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA -O .
cd ../configs
wget -O w32_256x192_adam_lr1e-3.yaml https://raw.githubusercontent.com/HRNet/HRNet-Human-Pose-Estimation/master/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
wget -O w32_384x288_adam_lr1e-3.yaml https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml
wget -O w48_256x192_adam_lr1e-3.yaml https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml
wget -O w48_384x288_adam_lr1e-3.yaml https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml