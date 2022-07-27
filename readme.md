# **USON:面向社交网络的情感分析**

## **简介**

一种可计算社交网络用户情感倾向的算法，并且可以融合用户情感倾向和文本特征进行情感分析

## **1 环境准备**

numpy 1.14.5

python 3.6.13

scikit-learn 0.20.1

sentencepiece 0.1.95

tensorflow 1.13.1

pytorch 1.9.1

## **2 数据集**

>data/amazon/data/train.tsv

amazon的训练集，包含标签，文本，用户ID

>data/amazon/data/test.tsv

amazon的测试集，包含标签，文本，用户ID

>data/amazon/amazon_user.csv

amazon用户信息数据，包括用户ID，好友数量，积极文本数量，消极文本数量

>data/amazon/amazonfriends.txt

amazon用户的好友关系表

>data/yelp/data/train.tsv

yelp的训练集，包含标签，文本，用户ID

>data/yelp/data/test.tsv

yelp的测试集，包含标签，文本，用户ID

>data/yelp/yelp_user.csv

yelp用户信息数据，包括用户ID，好友数量，积极文本数量，消极文本数量

>data/yelp/yelpfriends.txt

yelp用户的好友关系表

## **3 运行**

### 获得文本嵌入

对数据集分别学习，得到文本的句向量，以yelp为例，下面代码表示得到yelp训练集序列长度512下的文本嵌入

```
python extract_features.py
    --input_file=data/yelp/data/train.tsv
    --output_file=emb_data/yelp/emb_512/train_emb_512.csv
    --layers=-1
    --bert_config_file=config/config.json
    --init_checkpoint=yelp_bert_model/model.ckpt-8481
    --vocab_file=config/vocab.txt
    --max_seq_length=250
    --batch_size=8
```

--input_file 输入文件路径

--output_file 输出文件路径

--layers 取得文本嵌入的层数，-1，-2，-3，-4分别代表倒数第1，2，3，4层

--bert_config_file bert参数文件

--init_checkpoint 训练模型文件，附[已训练模型](https://pan.baidu.com/s/12a1zhteVgVUnIkJDH2lifg?pwd=msmc)

--vocab_file 语料库

--max_seq_length 文本序列长度

--batch_size 训练条数

### 获得用户嵌入

获取指定维度的用户嵌入，如下表示获取768维的yelp用户嵌入,会在emb_data/yelp/文件夹下生成yelp_user_embedding_768.txt

```
python pygcn/train.py
    --category=yelp
    --nemb=768
```

--category 指定数据集

--nemb 嵌入维度

### 融合文本嵌入和用户嵌入得到联合嵌入

将文本嵌入和用户嵌入融合得到联合嵌入，如下表示得到yelp句子序列长度512，用户嵌入768的联合嵌入，会在emb_data/yelp/生成文件夹sen_512_user_768，文件夹下包含train.tsv,dev.tsv,test.tsv

```
python social_fix.py
    --category=yelp
    --sen_len=512
    --user_emb=768
```

--category 数据集

--sen_len 句子序列长度

--user_emb 用户嵌入维度

### 训练&预测

对联合嵌入训练再预测测试集，如下是对yelp数据集训练预测，得到训练模型在tmp/fix_model文件夹下，预测结果在tmp/test_results.tsv

```
python fix_emb_classifier.py
    --data_dir=emb_data/yelp/sen_512_user_768
    --user_emb_length=768
    --num_train_epochs=30.0
```

--data_dir 联合嵌入数据集的路径，附已训练嵌入文件[amazon](https://pan.baidu.com/s/1zQa0RZNUjfl8zRhSIzK2Sw?pwd=nmsa)，[yelp](https://pan.baidu.com/s/1vixskxZvyzg-OroFDMsf1Q?pwd=s80z)

--user_emb_length 用户嵌入维度

-num_train_epochs 训练次数

## 验证结果

用f1指标验证测试结果，如下验证yelp的预测结果

```
python testf.py
    --category=yelp
```

--category 数据集类型


