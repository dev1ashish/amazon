import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from torchvision.models import resnet152, ResNet152_Weights
import timm

class MAG(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim):
        super(MAG, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion = nn.Linear(hidden_dim, text_dim)
        self.layer_norm = nn.LayerNorm(text_dim)

    def forward(self, text, img):
        text_proj = self.text_proj(text)
        img_proj = self.img_proj(img)
        gate = torch.sigmoid(self.gate(torch.cat([text_proj, img_proj], dim=-1)))
        fused = self.fusion(gate * img_proj)
        return self.layer_norm(text + fused)

class MXT(nn.Module):
    def __init__(self, text_model, img_model, xception_model, num_units):
        super(MXT, self).__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.xception_model = xception_model
        self.xception_projector = nn.Linear(2048, 768)
        
        self.mag = MAG(self.text_model.config.d_model, 2048, 512)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_model.config.d_model,
            num_heads=8,
            batch_first=True
        )
        
        self.value_regressor = nn.Sequential(
            nn.Linear(self.text_model.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.unit_classifier = nn.Linear(self.text_model.config.d_model, num_units)

    def forward(self, input_ids, attention_mask, image, allowed_units):
        # Text encoding
        encoder_outputs = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_embeds = encoder_outputs.last_hidden_state

        # Image encoding with ResNet
        img_features = self.img_model(image)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten the features
        img_features = img_features.unsqueeze(1).repeat(1, text_embeds.size(1), 1)
        
        # Apply MAG
        fused_embeds = self.mag(text_embeds, img_features)
        
        # Image encoding with Xception
        xception_features = self.xception_model(image)
        xception_features = xception_features.view(xception_features.size(0), -1, 2048)
        xception_features = self.xception_projector(xception_features)
        xception_features = xception_features.expand(-1, fused_embeds.size(1), -1)

        # Cross-attention
        attn_output, _ = self.cross_attention(fused_embeds, xception_features, xception_features)
        
        # Decode
        decoder_outputs = self.text_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=attn_output,
            return_dict=True
        )
        
        decoder_last_hidden_state = decoder_outputs.last_hidden_state
        
        # Predict value and unit
        value_output = self.value_regressor(decoder_last_hidden_state).squeeze(-1)
        unit_logits = self.unit_classifier(decoder_last_hidden_state)
        unit_logits = unit_logits.masked_fill(~allowed_units.unsqueeze(1).bool(), float('-inf'))
        
        return value_output, unit_logits

def load_mxt_model(device, num_units):
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
    xception = timm.create_model('xception', pretrained=True, num_classes=0)
    
    # Remove the last classification layer from ResNet
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    
    return MXT(t5_model, resnet, xception, num_units).to(device)