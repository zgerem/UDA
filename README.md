# UDA for Semantic Segmentation
The project focuses on the doman adaptation for semantic segmentation scenario GTA5 -> Cityscapes.
Overall training consists of training two models. The first model will be trained to get pseudo labels. Using these pseudo labels, the second model will be trained.
<div align="center">
<img width="796" alt="Ekran Resmi 2021-08-23 02 05 29" src="https://user-images.githubusercontent.com/56236171/130417258-d1b7b0d0-e810-48be-9764-915e616439b0.png">
</div>

## Training for pseudo labels
The first model is trained by running the notebook `training_psuedolabel_backbone.ipynb`. Then, running `evaluation_pseudo.ipynb` will save the pseudo labels. 

## Training with pseudo labels
Training of the second model consists of two steps. In the first step, the model will be initialized and trained with source images in target style by running the notebook `training_contrastive_step1.ipynb`. 
In the second step, target images and pseudo labels will be used for training by running the notebook `training_contrastive_step1.ipynb`. In this step, the model will be initialized with the weights of the first step. Running `evaluation.ipynb` will give the results on validation data. 

Code is adapted from [FDA](https://github.com/YanchaoYang/FDA)
