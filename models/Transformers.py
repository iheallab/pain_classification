import math
from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# this code is based on https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

class IMUTransformer(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")
        self.clinical_data_dim = config.get("clinical_data_dim")
        self.dropout_pos_enc = 0.1

        self.encoder_input_layer = nn.Linear(
            in_features=3,
            out_features=self.transformer_dim
        )

        # Create positional encoder
        # self.positional_encoding_layer = PositionalEncoder(
        #     d_model=self.transformer_dim,
        #     dropout=self.dropout_pos_enc,
        #     max_seq_len=10000
        # )
        self.positional_encoding_layer = PositionalEncoding(d_model=self.transformer_dim,
                                                            max_len=90002)

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
                                                nhead=config.get("nhead"),
                                                dim_feedforward=config.get("dim_feedforward"),
                                                dropout=config.get("transformer_dropout"),
                                                activation=config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        self.clin_embbeding = nn.Sequential(
                    nn.Linear(self.clinical_data_dim, self.transformer_dim),
        )
        #
        # if self.encode_position:
        #     self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes =  config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
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
        src = self.encoder_input_layer(inp1.permute(2, 0, 1)) #src shape: [batch_size, src length, dim_val] regardless of number of input features
        clin = self.clin_embbeding(inp2).unsqueeze(0)
        src = torch.cat([src, clin])
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        # if self.encode_position:
        #     src += self.position_embed
        src = self.positional_encoding_layer(
            src)

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
        out = self.imu_head(target)
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

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        #>>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            #>>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# class PositionalEncoder(nn.Module):
#     """
#     The authors of the original transformer paper describe very succinctly what
#     the positional encoding layer does and why it is needed:
#
#     "Since our model contains no recurrence and no convolution, in order for the
#     model to make use of the order of the sequence, we must inject some
#     information about the relative or absolute position of the tokens in the
#     sequence." (Vaswani et al, 2017)
#     Adapted from:
#     https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#     """
#
#     def __init__(
#             self,
#             dropout: float = 0.1,
#             max_seq_len: int = 5000,
#             d_model: int = 512,
#             batch_first: bool = False
#     ):
#         """
#         Parameters:
#             dropout: the dropout rate
#             max_seq_len: the maximum length of the input sequences
#             d_model: The dimension of the output of sub-layers in the model
#                      (Vaswani et al, 2017)
#         """
#
#         super().__init__()
#
#         self.d_model = d_model
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         self.batch_first = batch_first
#
#         self.x_dim = 1 if batch_first else 0
#
#         # copy pasted from PyTorch tutorial
#         position = torch.arange(max_seq_len).unsqueeze(1)
#
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#
#         pe = torch.zeros(max_seq_len, 1, d_model)
#
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
#                [enc_seq_len, batch_size, dim_val]
#         """
#
#         x = x + self.pe[:x.size(self.x_dim)]
#
#         return self.dropout(x)