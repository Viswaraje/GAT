import torch
import time
from model import *
from data_process import process
import argparse
from my_utiils import *
from data_load import dataload

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--alph', dest='alph', type=float, default=0.30, help='')
parser.add_argument('--beta', dest='beta', type=float, default=0.30, help='')
parser.add_argument('--epoch', dest='epoch', type=int, default=350, help='')
parser.add_argument('--hidden_channels', dest='hidden_channels', type=int, default=256, help='')
parser.add_argument('--output_channels', dest='output_channels', type=int, default=100, help='')
parser.add_argument('--heads', dest='heads', type=int, default=4, help='Number of attention heads in GAT')

args = parser.parse_args()
start_time = time.time()

# ------ Data files
Drug_info_file = '../data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
IC50_threds_file = '../data/Drug/drug_threshold.txt'
Drug_feature_file = '../data/Drug/drug_graph_feat'
Cell_line_info_file = '../data/Celline/Cell_lines_annotations.txt'
Genomic_mutation_file = '../data/Celline/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '../data/Celline/GDSC_IC50.csv'
Gene_expression_file = '../data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'

# ------- Load data
drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs = dataload(
    Drug_info_file, IC50_threds_file, Drug_feature_file, Cell_line_info_file, Genomic_mutation_file,
    Cancer_response_exp_file, Gene_expression_file, Methylation_file
)

# ------- Split train and test sets
drug_set, cellline_set, train_edge, label_pos, train_mask, test_mask, atom_shape = process(
    drug_feature, mutation_feature, gexpr_feature, methylation_feature, data_new, nb_celllines, nb_drugs
)



# ------ Initialize Model with GAT
model = GraphCDR(
    hidden_channels=args.hidden_channels,
    encoder=Encoder(args.output_channels, args.hidden_channels, heads=args.heads),
    summary=Summary(args.output_channels, args.hidden_channels),
    feat=NodeRepresentation(atom_shape, gexpr_feature.shape[-1], methylation_feature.shape[-1], args.output_channels),
    index=nb_celllines
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
myloss = nn.BCELoss()

# ------ Training Function
def train():
    model.train()
    loss_temp = 0
    for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
        optimizer.zero_grad()
        pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
            drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
        )
        dgi_pos = model.loss(pos_z, neg_z, summary_pos)
        dgi_neg = model.loss(neg_z, pos_z, summary_neg)
        pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
        loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
        loss.backward()
        optimizer.step()
        loss_temp += loss.item()
    print('Train Loss:', round(loss_temp, 4))

# ------ Testing Function
def test():
    model.eval()
    with torch.no_grad():
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            _, _, _, _, pre_adj = model(
                drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
            )
            loss_temp = myloss(pre_adj[test_mask], label_pos[test_mask])
        yp = pre_adj[test_mask].detach().numpy()
        ytest = label_pos[test_mask].detach().numpy()
        AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
        print(f'Test Loss: {round(loss_temp.item(), 4)}')
        print(f'Test AUC: {round(AUC, 4)} | Test AUPR: {round(AUPR, 4)} | Test F1: {round(F1, 4)} | Test ACC: {round(ACC, 4)}')
    return AUC, AUPR, F1, ACC

# ------ Main Training Loop
final_AUC, final_AUPR, final_F1, final_ACC = 0, 0, 0, 0
for epoch in range(args.epoch):
    print(f'\nEpoch: {epoch}')
    train()
    AUC, AUPR, F1, ACC = test()
    if AUC > final_AUC:
        final_AUC, final_AUPR, final_F1, final_ACC = AUC, AUPR, F1, ACC

# ------ Final Results
elapsed = time.time() - start_time
print('---------------------------------------')
print(f'Elapsed time: {round(elapsed, 4)}s')
print(f'Final AUC: {round(final_AUC, 4)} | Final AUPR: {round(final_AUPR, 4)} | Final F1: {round(final_F1, 4)} | Final ACC: {round(final_ACC, 4)}')
print('---------------------------------------')
