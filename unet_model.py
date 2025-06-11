import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_blocks import AttentionGate, SelfAttention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, channels, dropout_rate=0.3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = 1
        for out_ch in channels:
            self.blocks.append(ConvBlock(in_ch, out_ch, dropout_rate))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
    
    def forward(self, x):
        features = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            features.append(x)
            x = pool(x)
        return features

class Decoder(nn.Module):
    def __init__(self, channels, attention_channels=256):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        channels = channels[::-1]
        for i in range(len(channels) - 1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, stride=2))
            self.attention_gates.append(AttentionGate(channels[i+1], channels[i+1], attention_channels))
            self.blocks.append(ConvBlock(channels[i], channels[i+1]))
    
    def forward(self, x, encoder_features):
        encoder_features = encoder_features[::-1]
        
        for i, (upconv, attn_gate, block) in enumerate(zip(self.upconvs, self.attention_gates, self.blocks)):
            x = upconv(x)
            skip = encoder_features[i+1]
            skip = attn_gate(skip, x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(out_channels, out_channels//2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels//2)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_conv = nn.Conv2d(out_channels//2, out_channels//2, 1)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        global_context = self.global_pool(x)
        global_context = self.context_conv(global_context)
        global_context = global_context.expand_as(x)
        
        x = self.activation(self.bn3(self.conv3(x)))
        x = x + global_context
        
        return x

class UNetWithAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        encoder_channels = model_config['encoder_channels']
        decoder_channels = model_config['decoder_channels']
        attention_channels = model_config['attention_channels']
        dropout_rate = model_config['dropout_rate']
        
        self.encoder = Encoder(encoder_channels, dropout_rate)
        self.bottleneck = Bottleneck(encoder_channels[-1], encoder_channels[-1]*2)
        self.decoder = Decoder([encoder_channels[-1]*2] + decoder_channels, attention_channels)
        self.self_attention = SelfAttention(decoder_channels[-1])
        
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], 1, 1)
        self.bbox_head = nn.Conv2d(decoder_channels[-1], 4, 1)
        self.confidence_head = nn.Conv2d(decoder_channels[-1], 1, 1)
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        
        bottleneck_out = self.bottleneck(encoder_features[-1])
        
        decoder_out = self.decoder(bottleneck_out, encoder_features)
        
        attended_features = self.self_attention(decoder_out)
        
        segmentation = torch.sigmoid(self.segmentation_head(attended_features))
        bbox_regression = F.relu(self.bbox_head(attended_features))
        confidence = torch.sigmoid(self.confidence_head(attended_features))
        
        return {
            'segmentation': segmentation,
            'bbox_regression': bbox_regression,
            'confidence': confidence
        }