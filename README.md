
## This is a Repo for our Convolutional Visiton Transformer (CvT) derived project for 11785 DL Class

# Introduction
Our goal is to improve  CvT for classification problem to improve accuracy of classification task on MC-COCO dataset

## Innovation 
1. Multi-Scale Feature Extraction
2. Enhance positional Encoding and Embeddings
3. Deformable convolution and attention
4. Convolutional Self-attention
## Baseline
We successfully loaded the pretrained CvT-13 and CvT-21 to test on  ImageNet 1k Validation dataset. We get baseline result very similar to the paper's description:
| **Model** | **Input Resolution** | **Top-1 Accuracy (Replicated)** | **Top-1 Accuracy (Claimed)** |
|-----------|-----------------------|---------------------------------|------------------------------|
| CvT-13    | 224x224               | 81.626%                         | 81.6%                        |
| CvT-13    | 384x384               | 83.012%                         | 83.0%                        |
| CvT-21    | 224x224               | 82.523%                         | 82.5%                        |
| CvT-21    | 384x384               | 83.374%                         | 83.3%                        |

## Evaluation Metrics: 
1. FLOPs
2. Accuracy
3. Param size 




