from __future__ import division
from __future__ import print_function

import csv
import os.path
import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from pygcn.utils import load_data
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="yelp", help="Select the dataset")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--nemb', type=int, default=768,
                    help='Dims of user.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, name_features, labels, idx_test = load_data(args.category)
ulist = []
features = []
for key, value in name_features.items():
    ulist.append(key)
    features.append(value)
features = np.array(features)
labels = labels.reshape(len(labels), 1)

criterion = torch.nn.MSELoss()
# Model and optimizer
model = GCN(nfeat=features.shape[0],
            nemb=args.nemb,
            nhid=args.hidden,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

cuda0=torch.device("cpu")
model.to(cuda0)
features = torch.LongTensor(features)
features = features.to(cuda0)
adj = adj.to(cuda0)
labels = labels.to(cuda0)
# idx_train = idx_train.to(cuda0)
# idx_val = idx_val.to(cuda0)
# idx_test = idx_test.to(cuda0)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, user_embedding = model(features, adj)
    loss_train = criterion(output, labels)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, user_embedding = model(features, adj)
        loss_val = criterion(output[idx_test], labels[idx_test])
        if epoch % 100 == 0:
            print('Epoch: {:06d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    return loss_val, user_embedding,output


def test():
    model.eval()
    output, user_embedding = model(features, adj)
    loss_test = criterion(output[:, 0][idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
    user_embedding = user_embedding.detach().numpy()


# Train model
t_total = time.time()
e = 0
for epoch in range(args.epochs):
    loss_val, user_embedding ,output= train(epoch)
    if epoch == 0:
        loss_val_min = loss_val
    else:
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            print("best loss_val is:",loss_val_min.item())
            user_emb = user_embedding
            e = 0
        else:
            e += 1
            print("e=",e)
    if e > 1000:
        output_emb = output.tolist()
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
with open(os.path.join("emb_data", args.category, args.category+"_user_embedding_"+args.nemb+".txt"), "w") as f:
    f.write(str(len(features)) + " " + str(args.hidden) + "\n")
    for (name, embedding) in zip(name_features.keys(), user_emb):
        s = name + '\t' + ' '.join(str(f) for f in embedding.tolist())
        f.write(s + "\n")

with open('../data/user_output_emb.csv','w',encoding='utf-8',newline='') as f:
    csv_writer=csv.writer(f)
    csv_writer.writerows(output_emb)
# Testing
# test()
