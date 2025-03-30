from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

class SGGATConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 heads: int = 1, concat: bool = True, dropout: float = 0.6,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')  
        super(SGGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K  
        self.heads = heads  
        self.concat = concat  
        self.dropout = dropout  

        self.lin = Linear(in_channels, out_channels * heads, bias=False)
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))  
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels * heads))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)  

        for _ in range(self.K):
            x = self.propagate(edge_index, x=x)
        
        if self.concat:
            x = x.view(-1, self.heads * self.out_channels)  
        else:
            x = x.mean(dim=1)  

        if self.bias is not None:
            x = x + self.bias

        return x

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, ptr: OptTensor) -> Tensor:
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.exp(alpha)  
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)  
        return x_j * alpha.view(-1, self.heads, 1)  

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        return torch_scatter.scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self):
        return '{}({}, {}, heads={}, K={})'.format(self.__class__.__name__,
                                                   self.in_channels, self.out_channels,
                                                   self.heads, self.K)
