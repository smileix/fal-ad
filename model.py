import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import single


class AttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, D)
            mask: (optional) Bool tensor of shape (B, T), where True = keep, False = ignore
        Returns:
            Pooled tensor of shape (B, D)
        """
        # Compute raw attention scores
        scores = self.attn(x).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))  # mask out padding tokens

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, D)
        return pooled
    

class GatedAttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        attn_score = self.attn(x).squeeze(-1) # (B, T)
        gate_score = torch.sigmoid(self.gate(x)).squeeze(-1) # (B, T)
        scores = attn_score * gate_score  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # Shortcut
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNetAudio(nn.Module):
    def __init__(self, in_channels=1, out_channels=768, dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, dropout)
        self.layer2 = ResidualBlock(64, 128, dropout)
        self.layer3 = ResidualBlock(128, out_channels, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))  # Keep time dim, pool freq

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, F) — time, frequency (e.g., Mel spectrogram)
        Returns:
            Tensor of shape (B, T, out_channels)
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)  # (B, C, T, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, T, C)
        return x


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        """Cross-Attention Transformer Encoder Layer."""
        super(CrossAttentionEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):

        # Pre-Normalization
        src = self.norm1(src)
        memory = self.norm1(memory)

        # todo 后续可以尝试修改这个，在使用交叉注意力的时候，qkv尝试使用三种不同模态的特征，文本图像与语音
        # Cross-Attention (Query: src, Key/Value: memory)
        attn_output, _ = self.cross_attention(src, memory, memory, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)  # Residual Connection

        # Feed-Forward Network
        src = self.norm2(src)
        src = src + self.dropout2(self.feedforward(src))  # Residual Connection

        return src
    


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        """Gated Residual Cross-Attention Fusion Layer."""
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        Cross attention: src queries memory.
        Args:
            src: Tensor (B, T, d_model) — query (e.g., audio)
            memory: Tensor (B, T, d_model) — key/value (e.g., text)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
        Returns:
            Tensor (B, T, d_model): Fused output
        """

        # Cross-attention with pre-norm
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)

        attn_output, _ = self.cross_attention(
            query=src_norm,
            key=memory_norm,
            value=memory_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )

        attn_output = self.dropout1(attn_output) # # (B, T, d_model)
        gate_input = torch.cat([src, attn_output], dim=-1)  # (B, T, 2*d_model)
        gate = self.gate(gate_input)  # (B, T, d_model)
        fused = src + gate * attn_output  # Gated residual connection

        # Feed-forward with pre-norm
        fused = self.norm2(fused)
        fused = fused + self.dropout2(self.feedforward(fused))  # Residual

        return fused

class CrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder with Cross-Attention."""
        super(CrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.config = config


        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)

        if 'gated' in config.fusion:
            self.layers = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # Add LayerNorm between layers
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling

        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer cross-attention transformer encoder."""
        
        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''

        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)


        # Iterate over layers with normalization in between
        for i, layer in enumerate(self.layers):
            src = layer(src, memory, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers):  # Apply normalization between layers
                src = self.norm_layers[i](src)
                src = self.dropout(src)

        # Pooling strategy
        if self.pooling == 'mean':
            src = src.mean(dim=1)
        elif self.pooling == 'cls':
            src = src[:, 0, :]
        elif 'attn' in self.pooling:
            src = self.attn_pooling(src, mask=mask)

        # return self.classifier(src), all_vq_loss
        return self.classifier(src)
    
class BidirectionalCrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder with Cross-Attention."""
        super(BidirectionalCrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        

        if 'gated' in config.fusion:
            self.layers_1 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

            self.layers_2 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])
        else:
            self.layers_1 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

            self.layers_2 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        
        # Add LayerNorm between layers
        self.norm_layers_1 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.norm_layers_2 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling
        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        init_mlp_size = config.hidden_size * 2 if 'concat' in self.fusion else config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(init_mlp_size),
            nn.Dropout(config.dropout),
            nn.Linear(init_mlp_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer cross-attention transformer encoder."""
        
        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''


        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)

        # Copy src into src1 tensor
        src1 = src.clone()
        memory1 = memory.clone()

        # First embeddings
        for i, layer in enumerate(self.layers_1):
            src1 = layer(src1, memory1, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_1):  # Apply normalization between layers
                src1 = self.norm_layers_1[i](src1)
                src1 = self.dropout(src1)
        
        # Second embeddings
        src2 = memory.clone()
        memory2 = src.clone()

        for i, layer in enumerate(self.layers_2):
            src2 = layer(src2, memory2, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_2):  # Apply normalization between layers
                src2 = self.norm_layers_2[i](src2)
                src2 = self.dropout(src2)

        # Fuse src1 and src2

        if 'concat' in self.fusion:
            src = torch.cat((src1, src2), dim=2)
        elif 'sum' in self.fusion:
            src = src1 + src2
        elif 'mul' in self.fusion:
            src = src1 * src2
        elif 'mean' in self.fusion:
            src = (src1 + src2) / 2
        else:
            src = src1 + src2
            
        # Pooling strategy
        if self.pooling == 'mean':
            src = src.mean(dim=1)
        elif self.pooling == 'cls':
            src = src[:, 0, :]
        elif 'attn' in self.pooling:
            src = self.attn_pooling(src, mask=mask)


        return self.classifier(src)


class ElementWiseFusionEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder."""
        super(ElementWiseFusionEncoder, self).__init__()

        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        hidden_size = config.hidden_size * 2 if self.fusion == 'concat' else config.hidden_size


        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling
        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

        
    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer transformer encoder."""

        src, memory = features

        audio_model = self.model_name.split('_')[1] if self.config.audio_model != '' else ''

        if 'mel' == audio_model or 'egemaps' == audio_model:
            src = self.mel_extractor(src)
        elif 'mel' == audio_model or 'egemaps' == audio_model:
            memory = self.mel_extractor(memory)

        if self.fusion == 'concat':
            features = torch.cat((src, memory), dim=2)
        elif self.fusion == 'selfattn':
            src = src.mean(dim=1)
            memory = memory.mean(dim=1)
            features = torch.stack((src, memory), dim=1)
        elif self.fusion == 'mean':
            features = (src + memory) / 2
        elif self.fusion == 'sum':
            features = src + memory
        elif self.fusion == 'mul':
            features = src * memory
        
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # Pooling strategy
        if self.pooling == 'mean':
            features = features.mean(dim=1)
        elif self.pooling == 'cls':
            features = features[:, 0, :]
        elif 'attn' in self.pooling:
            features = self.attn_pooling(features, mask=mask)
        return self.classifier(features)


class MyTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder."""
        super(MyTransformerEncoder, self).__init__()

        self.model_name = config.model_name
        self.single_modal = config.single_modal
        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            self.mel_extractor = ResNetAudio(in_channels=1, out_channels=config.hidden_size)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling

        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer transformer encoder."""
        if self.single_modal == 'text':
            features = features[1]
        elif self.single_modal == 'audio':
            features = features[0]

        if 'mel' in self.model_name or 'egemaps' in self.model_name:
            features = self.mel_extractor(features)
        
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)

        if self.pooling == 'mean':
            features = features.mean(dim=1)
        elif self.pooling == 'cls':
            features = features[:, 0, :]
        elif 'attn' in self.pooling:
            features = self.attn_pooling(features, mask=mask)

        return self.classifier(features)
