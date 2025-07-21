import os

import torch
import torch.nn as nn

from lightning.pytorch import LightningModule
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, SymmetricMeanAbsolutePercentageError

from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig, TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CLIPLoRATemporalModel(LightningModule):
    def __init__(self, args, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.save_hyperparameters()
        
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", 'k_proj', 'out_proj']
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_config)
        self.clip.text_model = get_peft_model(self.clip.text_model, lora_config)
        
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.proj = nn.Linear(512, 512)
        self.temporal_model = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=args.temporal_range, padding=args.temporal_range//2),
            nn.Conv1d(256, 256, kernel_size=args.temporal_range, padding=args.temporal_range//2),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 8)
        
        self.align_weight = 0.1
        self.lr = args.lr
        
        self.rmse = MeanSquaredError(squared=False).to(self.device)
        self.mae = MeanAbsoluteError().to(self.device)
        self.smape = SymmetricMeanAbsolutePercentageError().to(self.device)
    
    def forward(self, video_tensor, text_list):
        '''
        args
        video_tensor: [B, T, 3, H, W]
        text_list: list of str, length B
        '''
        B, T, C, H, W = video_tensor.shape
        video_tensor = video_tensor.view(B * T, C, H, W)
        
        image_outputs = self.clip.vision_model(video_tensor)
        image_embeds = self.clip.visual_projection(image_outputs.pooler_output)
        image_embeds = image_embeds.view(B, T, -1)
        
        text_inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        text_outputs = self.clip.text_model(**text_inputs)
        text_embeds = text_outputs.pooler_output
        
        image_embeds = self.proj(image_embeds)
        x = image_embeds.permute(0, 2, 1)
        out = self.temporal_model(x)
        preds = self.out(out.permute(0, 2, 1))
        
        return preds, image_embeds, text_embeds
    
    def training_step(self, batch, batch_idx): 
        filepath, video_tensor, text_list, _, targets = batch
        preds, image_embeds, text_embeds = self.forward(video_tensor, text_list)
        
        regression_loss = nn.functional.mse_loss(preds, targets)
        
        image_mean = image_embeds.mean(dim=1)
        cosine_sim = nn.functional.cosine_similarity(image_mean, text_embeds, dim=-1)
        alignment_loss = 1 - cosine_sim.mean()
        
        total_loss = regression_loss + alignment_loss
        
        self.log_dict({
            "train_loss": total_loss,
            "regression_loss": regression_loss,
            "alignment_loss": alignment_loss
        }, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        filepath, video_tensor, text_list, _, targets = batch
        preds, image_embeds, text_embeds = self.forward(video_tensor, text_list)
        
        preds = preds.contiguous()
        rmse = self.rmse(preds, targets)
        mae = self.mae(preds, targets)
        smape = self.smape(preds, targets)
        
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log('val_mae', mae, prog_bar=True, sync_dist=True)
        self.log('val_smape', smape, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)