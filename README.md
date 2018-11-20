## notebook中包含数据预处理函数、特征提取函数、XGBoost函数以及F1score评价函数。

特征分三种：统计特征、语义特征、主题特征。

这里介绍统计特征仅供参考。
1.句子个数。2.单词拼写错误个数。3.错误标点个数。4.每句首字母是否大写。5.文章是否写完。6.句子长度统计量：最大长度、最小长度、平均长度。7.字数限制。8.单词个数。9.题目类型。10.用词丰富度。11.词性统计

注：9分类问题。样本共10000+，数据较不平衡，F1值为0.61。真实值与预测值之间得波动幅度较小（e.g. True4，Pred3），因此具有一定得参考性。

所需函数包:
import os<br>
import re<br>
import nltk<br>
import math<br>
import gensim<br>
import logging<br>
import enchant<br>
import nltk.data<br>
import numpy as np<br>
import pandas as pd<br>
import xgboost as xgb<br>
import multiprocessing<br>
from sklearn import metrics<br>
from collections import Counter<br>
from nltk.corpus import stopwords<br>
from gensim.models import LdaModel<br>
from gensim.models import Word2Vec<br>
from gensim import corpora, models <br>
from gensim.corpora import WikiCorpus<br>
from nltk.tokenize import WordPunctTokenizer<br>
from gensim.models.word2vec import LineSentence<br>
from sklearn.cross_validation import train_test_split<br>
logging.basicConfig(level=logging.DEBUG,  format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s') <br>
