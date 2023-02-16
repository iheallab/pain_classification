"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CNN_Transf_Skip(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super(CNN_Transf_Skip).__init__()

        self.transformer_dim = config.get("transformer_dim")
        self.clinical_data_dim = config.get("clinical_data_dim")

        self.input_proj = nn.Sequential(
            nn.Conv1d(config.get("input_dim"), self.transformer_dim//16, (1,)),
            nn.Conv1d(self.transformer_dim//16, self.transformer_dim//8, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(self.transformer_dim//8, self.transformer_dim//8, (1,)),
            nn.Conv1d(self.transformer_dim//8, self.transformer_dim//4, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(self.transformer_dim//4, self.transformer_dim//4, (1,)),
            nn.Conv1d(self.transformer_dim//4, self.transformer_dim//4, (1,)),
            nn.Conv1d(self.transformer_dim//4, self.transformer_dim//2, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(self.transformer_dim//2, self.transformer_dim//2, (1,)),
            nn.Conv1d(self.transformer_dim//2, self.transformer_dim//2, (1,)),
            nn.Conv1d(self.transformer_dim//2, self.transformer_dim, (1,)),
            nn.MaxPool1d(2),
            nn.GELU(),

            nn.Conv1d(self.transformer_dim, self.transformer_dim, (1,)),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, (1,)),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, (1,)),
            nn.MaxPool1d(2),
            nn.GELU()
            )

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        self.clin_embbeding = nn.Sequential(
                    nn.Linear(self.clinical_data_dim, self.transformer_dim),
        )

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes =  config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
        )
        self.add_info = nn.Sequential(
            nn.Linear(23, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self.late_fusion = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        # Shape N x S x C with S = sequence length, N = batch size, C = channels
        inp1 = data["acc"]
        inp2 = data["clin"]
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        cnn_out = self.input_proj(inp1).permute(2, 0, 1)
        clin = self.clin_embbeding(inp2).unsqueeze(0)
        src = torch.cat([cnn_out, clin])
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        transf_out = self.transformer_encoder(src)[0]
        transf_out = torch.unsqueeze(transf_out, 0)
        target = torch.cat([cnn_out, transf_out])
        target = torch.mean(target, dim=0)
        out = self.imu_head(target)
        # add_info = self.add_info(torch.reshape(inp2, (-1, 1)))
        # concat = torch.cat([out, add_info], dim=1)

        # transf_out = self.imu_head(target)
        #add_info = self.add_info(inp2)
        # concat = torch.cat([transf_out, add_info], dim=1)
        # late_fusion = self.late_fusion(concat)
        # Class probability
        target = self.log_softmax(out)
        return target


def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))