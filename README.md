# UDA for Semantic Segmentation
The project focuses on the doman adaptation for semantic segmentation scenario GTA5 -> Cityscapes.
Overall training consists of training two models. The first model will be trained to get pseudo labels. Using these pseudo labels, the second model will be trained.
<div align="center">
<img width="796" alt="Ekran Resmi 2021-08-23 02 05 29" src="https://user-images.githubusercontent.com/56236171/130417258-d1b7b0d0-e810-48be-9764-915e616439b0.png">
</div>

## Training for pseudo labels
The first model is trained by running the notebook `training_psuedolabel_backbone.ipynb`. Then, running `evaluation_pseudo.ipynb` will save the pseudo labels. 

## Training with pseudo labels
While training the second model, we use pseudo labels for target set. This time, we use contrastive loss and entropy minimization on target data in addition to cross entropy loss on source images in target style. The model will be trained by running notebook `training_contrastive.ipynb`. Then running `evaluation.ipynb` will give the results on validation data. 

Code is adapted from [FDA](https://github.com/YanchaoYang/FDA)
