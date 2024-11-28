import torch
from torch import cuda
from torch.optim import Adam

torch.manual_seed(3)
torch.cuda.manual_seed(3)
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_fb15k

from tqdm.autonotebook import tqdm

# Load dataset
kg_train, _, _ = load_fb15k()

# Define some hyper-parameters for training
emb_dim = 1024
lr = 0.0004
n_epochs = 10
b_size = 32768
margin = 0.5

# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(margin)
device = 'cpu'

# Move everything to CUDA if available
if cuda.is_available():
    device = 'cuda:0'
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(kg_train)

if device == 'cpu':
  dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=None)
else:
  dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')  

iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)

        optimizer.zero_grad()

        # forward + backward + optimize
        pos, neg = model(h, t, r, n_h, n_t)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description(
        'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                              running_loss / len(dataloader)))

model.normalize_parameters()

