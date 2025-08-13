from torch import nn
from torchvision.ops.misc import MLP

class paramDecoderMLP(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1,output_dim=1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.mlp=MLP(c_input,hidden_channels=[hidden_dim,hidden_dim,output_dim],norm_layer=nn.LayerNorm,dropout=dropout,activation_layer=nn.ReLU)

    def forward(self,x,mask):
        # 只对有效位置进行预测
        # x: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len] - True表示有效位置
        output = self.mlp(x)  # [batch_size, seq_len, output_dim]
        
        # 将无效位置设为0
        if mask is not None:
            output = output.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return output

class paramDecoderTransformer(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.mlp=nn.TransformerEncoder(nn.TransformerEncoderLayer(c_input, 4, hidden_dim, dropout), 1)
    def forward(self,x,mask):
        return self.mlp(x)