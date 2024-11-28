import torch
import pandas as pd
import numpy as np
from torch.nn import MarginRankingLoss
from torch.optim import Adam

from fastkg import SparseTransE, SparseKGDataset, corrupt_batch_sparse
from fastkg.datasets import load_fb15k

torch.manual_seed(3)
torch.cuda.manual_seed(3)
# from tqdm.autonotebook import tqdm
from tqdm.auto import tqdm

emb_dim = 1024
lr = 0.0004
n_epochs = 10
b_size = 32768
margin = 0.5

print(f'Running TransE Model for embedding size = {emb_dim}')
print('Loading fb15k dataset...', end='')

# # From TorchKGE
from torchkge.utils.datasets import load_fb15k
kg_train, _, _ = load_fb15k()
train_dataset = list(kg_train)
import pandas as pd
df = pd.DataFrame(train_dataset, columns=['from', 'to', 'rel'])
print('Done')
print('Converting to sparse dataset...', end='')
dataset_sparse = SparseKGDataset(df, batch_size=b_size, shuffle=False, drop_last=False, normalize=False)

# From own source
# df = load_fb15k('train')
# print('Done')
# print('Converting to sparse dataset...', end='')
# dataset_sparse = SparseKGDataset(df, batch_size=b_size, shuffle=False, drop_last=False, normalize=True)


dataloader_sparse = torch.utils.data.DataLoader(dataset_sparse,
                            batch_size=1,
                            drop_last=False,
                            collate_fn=lambda x: x[0])
print('Done')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device detected:', device)

model_sparse = SparseTransE(dataset_sparse.n_ent, dataset_sparse.n_rel, emb_dim).to(device)
criterion_sparse = MarginRankingLoss(margin=margin, reduction='sum').to(device)
optimizer_sparse = Adam(model_sparse.parameters(), lr=lr, weight_decay=1e-5)

print('Training...')

loss_lin_alg = []
# adj_neg = corrupt_batch_sparse(dataset_sparse[0], dataset_sparse.n_ent, dataset_sparse.relation_statistics, manual_seed=3)
a1 = torch.ones(b_size).to(device)
for m in (pbar := tqdm(range(n_epochs), leave=True)):
  for adj_pos in dataloader_sparse:
    adj_neg = corrupt_batch_sparse(adj_pos, dataset_sparse.n_ent, dataset_sparse.relation_statistics, manual_seed=None)
    # adj_neg = corrupt_batch_sparse(adj_pos, dataset_sparse.n_ent, None, manual_seed=None)
    adj_pos = adj_pos.to(device)
    adj_neg = adj_neg.to(device)
    optimizer_sparse.zero_grad()
    a, b = model_sparse(adj_pos, adj_neg)
    # a, b = model_sparse(adj_pos, adj_pos)
    if len(a) != len(a1):
      c = criterion_sparse(a, b, torch.ones_like(a).to(device))
    else:
      c = criterion_sparse(a, b, a1)
    # print(c.item())
    c.backward()
    # print(self_weights.grad, c.grad)
    optimizer_sparse.step()
    l = c.item()
    loss_lin_alg += [l]
    # model_sparse.normalize_entity_weights()
    pbar.set_description(f"Current Loss: {l:.4f}")

model_sparse.normalize_entity_weights()

print('Final Training loss:', loss_lin_alg[-1])

