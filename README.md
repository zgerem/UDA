# UDA for Semantic Segmentation
Overall training consists of training two models in two steps: training for pseudo labels and training with pseudo labels.
## Training for pseudo labels
The first model is trained by running the notebook `training_psuedolabel_backbone.ipynb`. Then, running `evaluation_pseudo.py` will save the pseudo labels. 

## Training with pseudo labels
The second model is trained by running the notebook `training_contrastive_deeplab.ipynb`. Then, running `evaluation_single.py` will give the results on validation data. 
