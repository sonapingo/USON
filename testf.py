import csv
import argparse
import os
import scipy.stats
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="yelp", help="Select the dataset")
args = parser.parse_args()

with open(os.path.join('tmp', 'test_results.tsv')) as f:
    reader=csv.reader(f, delimiter='\t')
    prey = []
    for row in reader:
        row = [float(x) for x in row]
        y=row.index(max(row))
        prey.append(y)
with open(os.path.join('data', args.category, 'data/test.tsv'),encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    oldy = []
    for row in reader:
        y = int(row[0])
        oldy.append(y)
assert len(prey) == len(oldy)
n = len(prey)
TP = FP = TN = FN = 0
for i in range(n):
    if oldy[i] == 1 and prey[i] == 1:
        TP += 1
    elif oldy[i] == 0 and prey[i] == 1:
        FP += 1
    elif oldy[i] == 1 and prey[i] == 0:
        FN += 1
    else:
        TN += 1
pos_p = TP / (TP + FP)
pos_r = TP / (TP + FN)
pos_f1 = 2 * pos_p * pos_r / (pos_p + pos_r)
neg_p = TN / (TN + FN)
neg_r = TN / (TN + FP)
neg_f1 = 2 * neg_p * neg_r / (neg_p + neg_r)
f1 = (pos_f1 + neg_f1) / 2
pv=scipy.stats.ranksums(oldy, prey)
ma_f1=f1_score(oldy,prey,average='macro')
print('ma-f1:',ma_f1)
print('neg-f1:',neg_f1)
print('pos-f1:',pos_f1)
print('f1:',f1)
