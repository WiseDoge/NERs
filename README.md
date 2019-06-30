# NERs
Some basic NER algorithms common in both Chinese and English.
## Dependencies
* Python 3.6+
* PyTorch 0.4+
* NumPy
* Pandas  （友好显示表格数据）
* Matplotlib （画混淆矩阵）
* Seaborn （画混淆矩阵）
* Sklearn （计算P，R，F1）    

安装依赖  
```pip install -r requirements.txt```
## Dataset
### Resume Data
选自论文ACL 2018[Chinese NER using Lattice LSTM](https://github.com/jiesutd/LatticeLSTM)中收集的简历数据，数据的格式如下，它的每一行由一个字及其对应的标注组成，标注集采用BIOES，句子之间用一个空行隔开。

```
美	B-LOC
国	E-LOC
的	O
华	B-PER
莱	I-PER
士	E-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```
### CoNLL 2003
```
EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER

BRUSSELS NNP B-NP B-LOC
1996-08-22 CD I-NP O
```
### Your Own Dataset
请将你自己的数据集按照以下格式组织。

```
sent1word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag1
sent1word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag2
sent1word3 [sent1word1 attr1] ... [sent1word1 attr_n] sent1tag2

sent2word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent2tag1
sent2word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent2tag2

sent3word1 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag1
sent3word2 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag2
sent3word3 [sent1word1 attr1] ... [sent1word1 attr_n] sent3tag3
```
本项目只会读取第一列（句子）和最后一列（NER标注）。

## Usage
先到 `train.py` 选择要训练的模型。
```python
# train.py 第 28 行
# 对于不需要训练的模型，将其对应的行注释掉即可
taggers = [
        LRTagger(*params[:2], *params[3:]),
        HMMTagger(len(word_to_ix), tag_dim, tag_to_ix['[PAD]']),
        CNNTagger(*params),
        BiLSTMTagger(*params),
        BiLSTMCRFTagger(*mask_model_params),
        BiLSTMAttTagger(*mask_model_params),
        BiLSTMCNNTagger(*params),
        CNNBiLSTMTagger(*params),
        CNNBiLSTMAttTagger(*mask_model_params)
    ]
```

### 在中文简历数据上进行训练和评估（此脚本 Windows/Linux 通用）  
```$ resume_train_eval.bat``` 
### 在 CoNLL 2003 数据上进行训练和评估（此脚本 Windows/Linux 通用）  
```$ conll_train_eval.bat``` 
### 在自己的数据集上进行训练和评估  
* 生成词表   
```python main.py --do_init --train_file=* --dev_file=* --test_file=* --word_dict_path=* --tag_dict_path=*```
* 训练  
```python main.py --do_train --output_dir=* --train_file=* --word_dict_path=* --tag_dict_path=*```  
亦可自行指定学习率、epochs等参数。 
* 评估  
```python main.py --do_eval --test_file=* --word_dict_path=* --tag_dict_path=* --output_dir=* --eval_log_dir=*```  
  
标 `*` 的位置需要用户自行指定。

## Results
评估的方法是对每一种标签分别求 **P**、**R**和**F1**，然后再求加权平均（`O` 标记也计算在内），评估的结果保存在 `--eval_log_dir` 目录。每一个模型都有三个输出文件：各类别的PRF1，混淆矩阵表格和混淆矩阵图。除此之外还会输出一个总的表格，表格内汇总了各模型的加权PRF1。
### Chinese Resume Data
|                      | Precision | Recall | F1-Score |
| -------------------- | --------- | ------ | -------- |
| Logistic Regression  | 0.7544    | 0.7634 | 0.7557   |
| HMM                  | 0.9207    | 0.9015 | 0.9095   |
| CNN                  | 0.9160    | 0.9158 | 0.9153   |
| BiLSTM               | 0.9546    | 0.9544 | 0.9542   |
| BiLSTM+Attention     | 0.9578    | 0.9577 | 0.9576   |
| BiLSTM+CNN           | 0.9571    | 0.957  | 0.9569   |
| CNN+BiLSTM           | 0.9584    | 0.9583 | 0.9579   |
| CNN+BiLSTM+Attention | 0.9615    | 0.9613 | 0.9612   |
| BiLSTM+CRF           | 0.9573    | 0.9572 | 0.9569   |
### CoNLL2003 
|                    | Precision | Recall | F1-Score |
| ------------------ | --------- | ------ | -------- |
| LRTagger           | 0.8865    | 0.8759 | 0.8803   |
| HMMTagger          | 0.9547    | 0.7362 | 0.8231   |
| CNNTagger          | 0.9111    | 0.9139 | 0.9119   |
| BiLSTMTagger       | 0.9218    | 0.9236 | 0.922    |
| BiLSTMCRFTagger    | 0.9209    | 0.9202 | 0.9202   |
| BiLSTMAttTagger    | 0.9233    | 0.9279 | 0.9239   |
| BiLSTMCNNTagger    | 0.9255    | 0.9286 | 0.9264   |
| CNNBiLSTMTagger    | 0.9191    | 0.9217 | 0.9196   |
| CNNBiLSTMAttTagger | 0.9164    | 0.9206 | 0.9176   |

## Using Docker
1. Pull image  
```$ docker pull wisedoge/ners```
2. Run  
```$ docker run -it wisedoge/ners```

## TODO
* 增加 Ensemble。