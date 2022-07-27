# this model user_emb and sen_emb are fixed
# optimization from BERT

import csv
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="yelp", help="Select the dataset")
parser.add_argument("--sen_len", type=str, default="512", help="Select the dataset")
parser.add_argument("--user_emb", type=str, default="768", help="Select the dataset")
args = parser.parse_args()

train_sen_emb = []
test_sen_emb = []
train_u_emb = []
test_u_emb = []
train_label = []
test_label = []
u_emb = {}

with open(os.path.join("emb_data", args.category, args.category + "_user_embedding_" + args.user_emb + ".txt"),
          encoding='utf-8') as f:
    next(f)
    reader = f.readlines()
    for row in reader:
        row = row.strip().split("\t")
        user = row[0]
        vector = row[1].split()
        vector = [float(x) for x in vector]
        u_emb[user] = vector

with open(os.path.join("emb_data", args.category, "emb_" + args.sen_len, "train_emb_" + args.sen_len + ".csv"),
          encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = [float(x) for x in row]
        train_sen_emb.append(row)

with open(os.path.join("emb_data", args.category, "emb_" + args.sen_len, "test_emb_" + args.sen_len + ".csv"),
          encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = [float(x) for x in row]
        test_sen_emb.append(row)

with open(os.path.join("data", args.category, "data", "train.tsv"), encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        label = int(row[0])
        user = row[2]
        train_label.append(label)
        train_u_emb.append(u_emb[user])

with open(os.path.join("data", args.category, "data", "test.tsv"), encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        label = int(row[0])
        user = row[2]
        test_label.append(label)
        test_u_emb.append(u_emb[user])

print(len(train_label))
print(len(train_sen_emb))
assert len(train_label) == len(train_sen_emb)
assert len(test_label) == len(test_sen_emb)
print("test len = ", len(test_sen_emb))
print("load data done!")

train_file = []
test_file = []
for i in range(len(train_label)):
    label = train_label[i]
    sen_emb = train_sen_emb[i]
    u_emb = train_u_emb[i]
    cur_dict = {}
    cur_dict["label"] = label
    cur_dict["sen_emb"] = sen_emb
    cur_dict["u_emb"] = u_emb
    train_file.append([cur_dict])
random.shuffle(train_file)

if args.category == "amazon":
    divide = 7690
elif args.category == "yelp":
    divide = 11300
elif args.category == "yelpnew":
    divide = 4000

dev_file = train_file[:divide]
train_file = train_file[divide:]
for i in range(len(test_sen_emb)):
    label = test_label[i]
    sen_emb = test_sen_emb[i]
    u_emb = test_u_emb[i]
    cur_dict = {}
    cur_dict["label"] = label
    cur_dict["sen_emb"] = sen_emb
    cur_dict["u_emb"] = u_emb
    test_file.append([cur_dict])

if (os.path.exists(os.path.join("emb_data", args.category, "sen_" + args.sen_len + "_user_" + args.user_emb))==False):
    os.makedirs(os.path.join("emb_data", args.category, "sen_" + args.sen_len + "_user_" + args.user_emb))


with open(os.path.join("emb_data", args.category, "sen_" + args.sen_len + "_user_" + args.user_emb, "train_emb.tsv"),
          "w", encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(train_file)
with open(os.path.join("emb_data", args.category, "sen_" + args.sen_len + "_user_" + args.user_emb, "dev_emb.tsv"),
          "w", encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(dev_file)
with open(os.path.join("emb_data", args.category, "sen_" + args.sen_len + "_user_" + args.user_emb, "test_emb.tsv"),
          "w", encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(test_file)

print("Write data done!")
