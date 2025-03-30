import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool

from torch.nn import Parameter
from my_utiils import *
from torch_geometric.nn import GATConv  # Import GATConv
from my_utiils import *
from base_model.sggat_conv import SGGATConv

from torch_geometric.nn import  global_mean_pool, global_max_pool as gmp
EPS = 1e-15

class NodeRepresentation(nn.Module):
    def __init__(self, gcn_layer, dim_gexp, dim_methy, output, units_list=[256, 256, 256], heads=2, use_relu=True, use_bn=True,
                 use_GMP=True, use_mutation=True, use_gexpr=True, use_methylation=True):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)
        
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.units_list = units_list
        self.use_GMP = use_GMP
        self.use_mutation = use_mutation
        self.use_gexpr = use_gexpr
        self.use_methylation = use_methylation
        
        # -------Drug layers using SGGATConv (Graph Attention Network)
        self.conv1 = SGGATConv(gcn_layer, units_list[0], heads=heads)
        self.batch_conv1 = nn.BatchNorm1d(units_list[0] * heads)

        self.graph_conv = nn.ModuleList()
        self.graph_bn = nn.ModuleList()
        for i in range(len(units_list) - 1):
            self.graph_conv.append(SGGATConv(units_list[i] * heads, units_list[i + 1], heads=heads))
            self.graph_bn.append(nn.BatchNorm1d(units_list[i + 1] * heads))

        self.conv_end = SGGATConv(units_list[-1] * heads, output, heads=1)  # Single head for final output
        self.batch_end = nn.BatchNorm1d(output)
        
        # --------Cell line layers (three omics)
        # -------Gene Expression (GEXP)
        self.fc_gexp1 = nn.Linear(dim_gexp, 256)
        self.batch_gexp1 = nn.BatchNorm1d(256)
        self.fc_gexp2 = nn.Linear(256, output)

        # -------Methylation (METHY)
        self.fc_methy1 = nn.Linear(dim_methy, 256)
        self.batch_methy1 = nn.BatchNorm1d(256)
        self.fc_methy2 = nn.Linear(256, output)

        # -------Mutation Data (MUT)
        self.conv_mut1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.conv_mut2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.fla_mut = nn.Flatten()
        self.fc_mut = nn.Linear(2010, output)

        # ------Concatenation Layer for Three Omics
        self.fcat = nn.Linear(300, output)
        self.batchc = nn.BatchNorm1d(output)
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data):
        # -----Drug Representation using GAT layers
        x_drug = F.leaky_relu(self.batch_conv1(self.conv1(drug_feature, drug_adj)))
        for conv, bn in zip(self.graph_conv, self.graph_bn):
            x_drug = F.leaky_relu(bn(conv(x_drug, drug_adj)))
        x_drug = F.leaky_relu(self.batch_end(self.conv_end(x_drug, drug_adj)))
        
        # Graph Pooling
        if self.use_GMP:
            x_drug = gmp(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)

        # -----Cell Line Representation
        # -----Mutation Data Processing
        if self.use_mutation:
            x_mutation = torch.tanh(self.conv_mut1(mutation_data))
            x_mutation = F.max_pool2d(x_mutation, (1, 5))
            x_mutation = F.relu(self.conv_mut2(x_mutation))
            x_mutation = F.max_pool2d(x_mutation, (1, 10))
            x_mutation = self.fla_mut(x_mutation)
            x_mutation = F.relu(self.fc_mut(x_mutation))

        # -----Gene Expression Processing
        if self.use_gexpr:
            x_gexpr = torch.tanh(self.fc_gexp1(gexpr_data))
            x_gexpr = self.batch_gexp1(x_gexpr)
            x_gexpr = F.relu(self.fc_gexp2(x_gexpr))

        # -----Methylation Processing
        if self.use_methylation:
            x_methylation = torch.tanh(self.fc_methy1(methylation_data))
            x_methylation = self.batch_methy1(x_methylation)
            x_methylation = F.relu(self.fc_methy2(x_methylation))

        # ------Concatenate Omics Representations
        if not self.use_gexpr:
            x_cell = torch.cat((x_mutation, x_methylation), dim=1)
        elif not self.use_mutation:
            x_cell = torch.cat((x_gexpr, x_methylation), dim=1)
        elif not self.use_methylation:
            x_cell = torch.cat((x_mutation, x_gexpr), dim=1)
        else:
            x_cell = torch.cat((x_mutation, x_gexpr, x_methylation), dim=1)

        x_cell = F.relu(self.fcat(x_cell))

        # ------Combine Drug & Cell Line Representations
        x_all = torch.cat((x_cell, x_drug), dim=1)
        x_all = self.batchc(x_all)

        return x_all




class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.prelu1 = nn.PReLU(hidden_channels * heads)  # Adjusted for multiple heads

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)  # Apply GAT convolution
        x = self.prelu1(x)  # Apply PReLU activation
        return x  # Return transformed node embeddings


class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 1)

    def forward(self, xo, xn):
        m = self.fc1(torch.cat((xo, xn), 1))
        m = torch.tanh(torch.squeeze(m))
        m = torch.exp(m) / (torch.exp(m)).sum()
        x = torch.matmul(m, xn)
        return x

class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(GraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(100, 10)
        self.fd = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data, edge):
        # ---CDR_graph_edge and corrupted CDR_graph_edge
        if not isinstance(edge, torch.Tensor):
            edge = torch.tensor(edge)  # Convert NumPy array to PyTorch tensor

        pos_edge = edge[edge[:, 2] == 1, :2].T.contiguous().long()
        neg_edge = edge[edge[:, 2] == -1, :2].T.contiguous().long()
        
        # ---cell+drug node attributes
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, methylation_data)
        
        # ---cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        # ---graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        # ---embedding at layer l
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        # ---embedding at layer 0
        cellfea = self.fc(feature[:self.index, ])
        drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea)
        drugfea = torch.sigmoid(drugfea)
        
        # ---concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        
        # ---inner product
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value


    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(torch.clamp(self.discriminate(pos_z, summary, sigmoid=True), min=EPS)).mean()
        neg_loss = -torch.log(torch.clamp(1 - self.discriminate(neg_z, summary, sigmoid=True), min=EPS)).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
