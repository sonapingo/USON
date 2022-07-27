import numpy as np
import os
import scipy.sparse as sp
import torch
import csv


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    with open(os.path.join("data", dataset, dataset+"_user.csv"), encoding="utf-8") as f:
        reader=csv.reader(f,delimiter="\t")
        next(reader)
        name_label=[]
        for row in reader:
            name=row[0]
            p=int(row[2])
            n=int(row[3])
            label=(p-n)/(p+n)
            label=float("%.4f" % label)
            name_label.append([name,label])

    name_label=np.array(name_label)
    idx=np.array(name_label[:,0],dtype=np.str)
    labels=np.array(name_label[:,1],dtype=np.float)
    idx_map={j:i for i, j in enumerate(idx)}

    with open(os.path.join("data", dataset, dataset+"friends.txt"), encoding="utf-8") as f:
        reader=f.readlines()
        edges_unordered=[]
        for row in reader:
            row=row.strip().split()
            user_1=row[0]
            user_2=row[1:]
            for fri in user_2:
                if fri not in idx_map.keys():
                    continue
                edges_unordered.append([user_1,fri])
    with open(os.path.join("data", dataset, dataset+"friends.txt"), encoding="utf-8") as f:
        reader = f.readlines()
        b=0
        for row in reader:
            row = row.strip().split()
            user_2 = row[1:]
            c=len(user_2)
            if c>b:
                b=c

    edges_unordered=np.array(edges_unordered,dtype=np.str)
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    usersize=len(name_label)

    idx_train = range(usersize)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, idx_map, labels, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
