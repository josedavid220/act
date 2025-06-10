import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResBlock, Upsampler, default_conv

from .embedding import Embedding, unembedding
from .convolution import ResidualGroup, FB


class ActTime(torch.nn.Module):
    def __init__(
        self,
        args
    ):
        super(ActTime, self).__init__()

        conv = default_conv

        self.n_channels = args.n_channels
        self.n_feats = args.n_feats
        self.token_size = args.token_size
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.scale = args.scale
        self.reduction = args.reduction
        self.n_resblocks = args.n_resblocks
        self.n_resgroups = args.n_resgroups
        self.n_fusionblocks = args.n_fusionblocks
        self.expansion_ratio = args.expansion_ratio

        embedding_dim = self.n_feats * self.token_size
        hidden_dim = embedding_dim * self.expansion_ratio
        
        match args.act:
            case 'relu':
                act = nn.ReLU(True)
            case 'elu':
                act = nn.ELU(True)
            case 'gelu':
                act = nn.GELU(True)
            case _:
                act = nn.ReLU(True)                

        # Head definition
        self.head = nn.Sequential(
            default_conv(self.n_channels, self.n_feats, 3),
            ResBlock(conv=conv, n_feats=self.n_feats, kernel_size=5, act=act),
            ResBlock(conv=conv, n_feats=self.n_feats, kernel_size=5, act=act),
        )

        # Body definition
        # Transformer encoder and embedding layers
        self.embedding = Embedding(n_feats=self.n_feats, token_size=self.token_size)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=self.n_heads)
                for _ in range(self.n_layers // 2)
            ]
        )

        # Convolution layers with channel attetion
        modules_cnn = [
            ResidualGroup(
                conv=conv,
                n_feat=self.n_feats,
                kernel_size=3,
                reduction=self.reduction,
                n_resblocks=self.n_resblocks,
            )
            for _ in range(self.n_resgroups)
        ]

        modules_cnn.append(
            conv(in_channels=self.n_feats, out_channels=self.n_feats, kernel_size=3)
        )
        self.cnn_branch = nn.Sequential(*modules_cnn)

        # Fusion Blocks
        self.fusion_block = nn.ModuleList(
            [
                nn.Sequential(
                    FB(conv, self.n_feats * 2, kernel_size=1, act=act),
                    FB(conv, self.n_feats * 2, kernel_size=1, act=act),
                    FB(conv, self.n_feats * 2, kernel_size=1, act=act),
                    FB(conv, self.n_feats * 2, kernel_size=1, act=act),
                )
                for _ in range(self.n_fusionblocks)
            ]
        )

        # MLP stand for multilayer perceptron
        self.fusion_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                for _ in range(self.n_fusionblocks - 1)
            ]
        )

        self.fusion_cnn = nn.ModuleList(
            [
                nn.Sequential(
                    conv(
                        in_channels=self.n_feats,
                        out_channels=self.n_feats,
                        kernel_size=3,
                    ),
                    act,
                    conv(
                        in_channels=self.n_feats,
                        out_channels=self.n_feats,
                        kernel_size=3,
                    ),
                )
                for _ in range(self.n_fusionblocks - 1)
            ]
        )

        # Single convolution to lessen dimension after body module
        self.conv_last = conv(
            in_channels=self.n_feats * 2, out_channels=self.n_feats, kernel_size=3
        )

        # Tail definition
        self.tail = nn.Sequential(
            Upsampler(conv=conv, scale=self.scale, n_feats=self.n_feats, act=False),
            conv(in_channels=self.n_feats, out_channels=self.n_channels, kernel_size=3),
        )

    def forward(self, x):
        seq_len = x.size(-1)  # Save sequence length for later steps

        # x = F.layer_norm(x, normalized_shape=(seq_len,))
        x = x.unsqueeze(1)  # Add channel dimmension of 1
        x = self.head(x)
        identitiy = x  # Save head result for residual connection before tail

        x_tkn = self.embedding(x)  # Generate embeddings for transformer

        for i in range(self.n_fusionblocks):
            x_tkn = self.encoder_layers[i](x_tkn)  # Transformer block result
            x = self.cnn_branch[i](x)  # CNN block result

            x_tkn_res, x_res = (
                x_tkn,
                x,
            )  # Save output for residual connection after fusion
            x_tkn = unembedding(
                x_tkn, self.token_size, seq_len
            )  # Token unembedding for fusion

            f = torch.cat((x, x_tkn), 1)
            f = f + self.fusion_block[i](f)

            if i != (self.n_fusionblocks - 1):
                x_tkn, x = torch.split(f, self.n_feats, 1)

                x_tkn = self.embedding(x_tkn)
                x_tkn = self.fusion_mlp[i](x_tkn) + x_tkn_res

                x = self.fusion_cnn[i](x) + x_res
            break

        x = self.conv_last(f)
        x = x + identitiy  # Final residual connection
        x = self.tail(x).squeeze(1)  # Remove channel dimension

        return x
