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
        CNNTagger(*params[:3], max_seq_len, *params[3:]),
        BiLSTMTagger(*params),
        BiLSTMCRFTagger(*mask_model_params),
        BiLSTMAttTagger(*mask_model_params),
        BiLSTMCNNTagger(*params),
        CNNBiLSTMTagger(*params),
        CNNBiLSTMAttTagger(*mask_model_params)
    ]
```    

* 在中文简历数据上进行训练和评估（此脚本 Windows/Linux 通用）  
```$ resume_train_eval.bat``` 
* 在 CoNLL 2003 数据上进行训练和评估（此脚本 Windows/Linux 通用）  
```$ conll_train_eval.bat``` 
* 在自己的数据集上进行训练和评估
    * 生成词表   
```python main.py --do_init --train_file=* --dev_file=* --test_file=* --word_dict_path=* --tag_dict_path=*```
    * 训练  
```python main.py --do_train --output_dir=* --train_file=* --word_dict_path=* --tag_dict_path=*```  
亦可自行指定学习率、epochs等参数。 
    * 评估  
```python main.py --do_eval --test_file=* --word_dict_path=* --tag_dict_path=* --output_dir=* --eval_log_dir=*```  

    注：标 `*` 的位置需要用户自行指定。

## Training in Docker
1. Pull image  
```$ docker pull wisedoge/ners```
2. Run  
```$ docker run -it wisedoge/ners```

## Results
评估的方法是对每一种标签分别求 **P**、**R**和**F1**，然后再求加权平均（`O` 标记也计算在内），评估的结果保存在 `--eval_log_dir` 目录。每一个模型都有三个输出文件：各类别的PRF1，混淆矩阵表格和混淆矩阵图。除此之外还会输出一个总的表格，表格内汇总了各模型的加权PRF1。
### Chinese Resume Data
|              | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| LR           | 0.7578    | 0.7625 | 0.7568   |
| HMM          | 0.9207    | 0.9015 | 0.9095   |
| CNN          | 0.9161    | 0.9140 | 0.9144   |
| BiLSTM       | 0.9539    | 0.9537 | 0.9535   |
| BiLSTMCRF    | 0.9581    | 0.9579 | 0.9578   |
| BiLSTMAtt    | 0.9587    | 0.9583 | 0.9583   |
| BiLSTMCNN    | 0.9564    | 0.9565 | 0.9563   |
| CNNBiLSTM    | 0.9564    | 0.9562 | 0.9560   |
| CNNBiLSTMAtt | 0.9611    | 0.9611 | 0.9609   |
### CoNLL2003 
| Unnamed: 0   | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| LR           | 0.8844    | 0.8732 | 0.8778   |
| HMM          | 0.9543    | 0.7338 | 0.8211   |
| CNN          | 0.9136    | 0.9184 | 0.9148   |
| BiLSTM       | 0.9203    | 0.9196 | 0.9192   |
| BiLSTMCRF    | 0.9199    | 0.9226 | 0.9205   |
| BiLSTMAtt    | 0.9213    | 0.9243 | 0.9216   |
| BiLSTMCNN    | 0.9215    | 0.9255 | 0.9226   |
| CNNBiLSTM    | 0.9157    | 0.9188 | 0.9165   |
| CNNBiLSTMAtt | 0.9151    | 0.9191 | 0.9162   |

