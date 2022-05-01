from turtle import forward
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import ViTConfig, ViTModel, ViTFeatureExtractor


import pytorch_lightning as pl 

from src.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT_RATE
)

class BaseLine(pl.LightningModule):
    def __init__(
        self, 
        learning_rate: int = LEARNING_RATE,
        weight_decay: int = WEIGHT_DECAY,
        dropout_rate: int = DROPOUT_RATE, 

    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self.save_hyperparameters()

        # Define Architecture 
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


    
    def forward(self, imgs):
        ins = self.vit_feature_extractor(imgs, return_tensors="pt")
        outs = self.vit_model(**ins)

        return outs 

    def training_step(self, batch, batch_idx):
        
        imgs = batch[0]["image"]
        labels = batch[0]["label"]

        outs = self(imgs)

        import sys 
        print(outs)
        sys.exit()


    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        ) 

