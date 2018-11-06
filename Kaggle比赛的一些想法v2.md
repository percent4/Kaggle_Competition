建模流程：

1. 一些初步的统计结果，如问题数，pairs数量，单词数（语料库），相似问题的数量分布；
2. 文本预处理： 大小写，去掉标点，分词，去停用词，词形还原，同义词替换（WordNet）
3. 提取特征： TF-IDF， Word2Vec, 句子相似度（词袋模型）
4. 二分类问题的模型： RF, XGBoost, LSTM
5. 在test.csv文件上进行验证



1.是否可以利用WordNet可以寻找同义词，替换句子中相同意思的单词？

2.利用词袋模型计算句子间的相似度，考虑是否将其作为新的特征

```python
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')
from gensim import corpora, models, similarities
from nltk import word_tokenize
from gensim.similarities import Similarity
import numpy as np

df = pd.read_csv('F://Kaggle/train.csv').fillna("")

def string_similarity(df):

    if df['question1'] != "" and df['question2'] != "":
        sents = [df['question1'], df['question2']]
        #print(sents)

        # 分词
        texts = [[word for word in word_tokenize(document)] for document in sents]
        #print(texts)

        #  语料库
        dictionary = corpora.Dictionary(texts)

        # 利用doc2bow作为词袋模型
        corpus = [dictionary.doc2bow(text) for text in texts]
        #print(corpus)

        similarity = Similarity('-Similarity-index', corpus, num_features=len(dictionary))
        #print(similarity)

        new_sensence = df['question1']
        print(df['id'])
        test_corpus_1 = dictionary.doc2bow(word_tokenize(new_sensence))
        # print(test_corpus_1)
        # 获取新句子与文档中的每个句子的相似度
        t = similarity[test_corpus_1][1]
        #print(t)
        #print(df['is_duplicate'])

    else:
        t = 0

    # print(t)
    return t


similary_list = []

for i in range(df.shape[0]):
    print(i)
    similary_list.append(string_similarity(df.iloc[i, :]))

df['similariry'] = similary_list
print(df.head(n=10))
df.to_csv('F://Kaggle/train_with_similarity.csv')
```

对相似度进行分析

```python
import pandas as pd

df = pd.read_csv('F://Kaggle/train_with_similarity.csv').fillna("")

df0 = df[df['is_duplicate'] == 0]['similarity']
df1 = df[df['is_duplicate'] == 1]['similarity']

# 输出统计指标，总数，平均数，标准差，最小值，四分位数，最大值
similariy_0 = df0.describe()
similariy_1 = df1.describe()
stats_df = pd.DataFrame({'sim_0': similariy_0,
                         'sim_1': similariy_1})
print(stats_df)
```

输出结果：

```
               sim_0          sim_1
count  255027.000000  149263.000000
mean        0.458159       0.629157
std         0.243683       0.180023
min         0.000000       0.000000
25%         0.267261       0.500000
50%         0.426401       0.629941
75%         0.632456       0.769800
max         1.000000       1.000000
```

3. 数据预处理（对问题进行预处理：扩展缩写，转化为小写，分词，词形还原，去掉特殊符号，去掉停用词）

```python
# -*- coding: utf-8 -*-

import re
import nltk
import string
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer

# 停用词列表
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove("how")
stopword_list.remove("what")
stopword_list.remove("where")
stopword_list.remove("when")
stopword_list.remove("which")
stopword_list.remove("why")

# 词形还原
wnl = WordNetLemmatizer()

# 分词
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

# 扩展缩写
def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# 词性分析，使用POS tags
def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tokens = word_tokenize(text)
    tagged_text = pos_tag(tokens)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    return tagged_lower_text


# 词形还原
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

# 去掉特殊符号
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# 去掉停用词
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_text = [token for token in tokens if token not in stopword_list]
    return filtered_text

def string_preprocessing(question):
    print(question)
    try:
        if question:
            text = expand_contractions(question, CONTRACTION_MAP)
            text = text.lower()
            text = lemmatize_text(text)
            text = remove_special_characters(text)
            text = remove_stopwords(text)
        else:
            text = []

    except Exception:
        text = ["err"]

    return text

df = pd.read_csv('G://Kaggle/train.csv').fillna("")
df['question1_preprocessing'] = df['question1'].apply(string_preprocessing)
df['question2_preprocessing'] = df['question2'].apply(string_preprocessing)
print(df.head())
df.to_csv('G://Kaggle/train_after_preprocessing.csv')
```


4. 提取TF-IDF特征，word2vec

```python
# -*- coding: utf-8 -*-

import pandas as pd
import gensim
import numpy as np
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer

def str2list(text):
    text = text.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    if text:
        text = text.split(",")
    else:
        text = []

    return text

df = pd.read_csv('G://Kaggle/train_after_preprocessing.csv').fillna("")
#print(df.columns)

# tokens
ques1 = df['question1_preprocessing'].apply(str2list).tolist()
ques2 = df['question2_preprocessing'].apply(str2list).tolist()

# strings
feature_ques1 = [' '.join(item) for item in ques1]
feature_ques2 = [' '.join(item) for item in ques2]
#print(feature_ques1)

#  语料库
dictionary = corpora.Dictionary(ques1+ques2)
#for _ in dictionary.items():
    #print(_)

# TF-IDF特征
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 1))
feature_matrix1 = vectorizer.fit_transform(feature_ques1).astype(float)
feature_matrix2 = vectorizer.fit_transform(feature_ques1).astype(float)

#print(type(feature_matrix1))
#for i in range(feature_matrix1.shape[0]):
    #print(feature_matrix1[i])

# Word2Vec
output_size = 50
model = gensim.models.Word2Vec(ques1, size=output_size, window=5, min_count=1, sample=1e-3)
#print(model.wv.index2word)
#print(dictionary[list(dictionary.keys())[0]])
#print(model[dictionary[list(dictionary.keys())[0]]])

# 获取以TF-IDF为权重的word2vec向量
def get_word_vector(feature_matrix, model, dictionary):
    word_vector = []
    for row in feature_matrix:
        tf_idf_based_word2vec = np.array([0]*output_size, dtype=np.float64)
        if len(row.indices):
            for index in row.indices:
                tf_idf_based_word2vec += row[0, index]*np.array(model[dictionary[index]])
            tf_idf_based_word2vec = tf_idf_based_word2vec/len(row.indices)
        word_vector.append(tf_idf_based_word2vec)

    return np.array(word_vector)

word_vector1 = get_word_vector(feature_matrix1, model, dictionary)
word_vector2 = get_word_vector(feature_matrix2, model, dictionary)
#print(word_vector1)
#print(word_vector2)

df_data = {}
for i in range(output_size):
    df_data['v'+str(i+1)] = word_vector1[:, i]
for i in range(output_size):
    df_data['v'+str(i+output_size+1)] = word_vector2[:, i]

new_df = pd.DataFrame(df_data)

similarity_df = pd.read_csv('G://Kaggle/train_with_similarity.csv').fillna("")
new_df['similarity'] = similarity_df['similarity'].tolist()
new_df['label'] = df['is_duplicate'].tolist()

print(new_df.head())

new_df.to_csv("G://Kaggle/train_vector.csv", index=False)
```

5.使用XGBoost进行模型训练

```python
# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 读取数据，并分为特征和目标变量
df = pd.read_csv('G://Kaggle/train_vector.csv')

target_var = "label"
cols = list(df.columns)
cols.remove(target_var)
data= df[cols]
label = df[target_var]

# 将数据分为训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=0)

# 转化格式，便于xgboost调用
print("Using XGBoost model...")
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

# xgboost模型参数
params = {'booster':'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'max_depth':10,
          'lambda':10,
          'subsample':0.75,
          'colsample_bytree':0.75,
          'min_child_weight':2,
          'eta': 0.25,
          'seed':0,
          'nthread':4,
          'silent':1
          }

watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)

# 预测
ypred = bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1
#print(test_y, y_pred)
print('AUC: %.4f' % metrics.roc_auc_score(test_y,y_pred))
print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
```

输出结果：

```
......
[995]	train-auc:0.999843
[996]	train-auc:0.999843
[997]	train-auc:0.999845
[998]	train-auc:0.999845
[999]	train-auc:0.999846
AUC: 0.7824
ACC: 0.8055
Recall: 0.6941
F1-score: 0.7250
Precesion: 0.7589
```