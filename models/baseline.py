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
        # self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Freeze VIT 
        for param in self.vit_model.parameters():
            param.requires_grad = False


        self.encoder_net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(512, 3072),
            nn.ReLU()
        )

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=32),
            nn.MaxPool2d(kernel_size=3),
            nn.MaxPool2d(kernel_size=2)
        )

        self.loss_fn = nn.MSELoss()
    
    def forward(self, imgs):
        # ins = self.vit_feature_extractor(imgs, return_tensors="np")
        outs = self.vit_model(imgs)
        outs = self.encoder_net(outs.pooler_output)
        return outs 

    def training_step(self, batch, batch_idx):
        
        imgs = batch[0]["image"]
        # labels = batch[0]["label"]

        image_rep = self(imgs)

        upscaled_img = self.decoder_net(image_rep)
        flattened_imgs = self.conv_net(imgs)
        flattened_imgs = torch.flatten(flattened_imgs, start_dim=1)

        # import sys 
        # print(upscaled_img.size())
        # print(flattened_imgs.size())
        # sys.exit()

        loss = self.loss_fn(upscaled_img, flattened_imgs)

        self.log("loss/train", loss)

        return loss 
        
    def validation_step(self, batch, batch_idx):
        imgs = batch[0]["image"]
        # labels = batch[0]["label"]

        image_rep = self(imgs)

        upscaled_img = self.decoder_net(image_rep)
        flattened_imgs = self.conv_net(imgs)
        flattened_imgs = torch.flatten(flattened_imgs, start_dim=1)

        loss = self.loss_fn(upscaled_img, flattened_imgs)

        self.log("loss/val", loss)

        return loss 
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs = batch[0]["image"]
        # labels = batch[0]["label"]

        image_rep = self(imgs)

        return image_rep

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        ) 

