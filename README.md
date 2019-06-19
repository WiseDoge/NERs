# ChineseNER
Some Chinese named entity recognition(NER) algorithms.
## Dependencies
* Python 3.6+
* PyTorch 0.4+
* NumPy
* Pandas 
* Matplotlib
* Seaborn
* Sklearn
## Dataset

数据集用的是论文ACL 2018[Chinese NER using Lattice LSTM](https://github.com/jiesutd/LatticeLSTM)中收集的简历数据，数据的格式如下，它的每一行由一个字及其对应的标注组成，标注集采用BIOES，句子之间用一个空行隔开。

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
如果使用自己的数据集，只需将其格式化为上段所示的可是即可。

## Usage
* 修改配置文件 `config.py` (optional)
```python
OUTPUT_DIR = 'output'
DATA_DIR = 'data'
EVAL_LOG_DIR = 'evallog'
LEARNING_RATE = 0.001
PRINT_STEP = 20
......
```

* 生成词表
```
python initialize.py
```
* 训练并保存模型
```
python train.py
```
* 评估模型
```
python eval.py
```
## Results
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
| ~~CRF~~              |           |        |          |
| ~~BiLSTM-CRF~~       |           |        |          |
| ~~BiLSTM-CNN-CRF~~   |           |        |          |

## Getting Started
Waiting

## TODO
* 增加 CRF 模型及其与其他模型的混搭。
* 增加 Ensemble。