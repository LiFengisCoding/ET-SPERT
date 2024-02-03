# ET-SPERT
## This repository contains  part of the code for our paper
基于字节编码和预训练任务的加密流量分类模型

Note: this code is based on [UER-py](https://github.com/dbiir/UER-py) and [ET-BERT](https://github.com/linwhitehat/ET-BERT). Many thanks to the authors.

## Datasets
Sorry, due to protocol restrictions, the datasets and weights are not publicly available now.

## data_process
You can train it with the following command:
```
python data_process/data_preprocess.py
```

## Train
You can train it with the following command:
```
python pre-training/pretrain.py
```

## Evaluate
You can test it with the following command:
```
python fine-tuning/run_classifier.py
```

## Note
Unfortunately, due to protocol restrictions we cannot release the complete source code and models.
