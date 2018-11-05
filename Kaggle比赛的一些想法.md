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

#question = "Why am I mentally very lonely? How can I solve it?"
#question = "What is the story of Kohinoor (Koh-i-Noor) Diamond?"
#question = "How can I increase the speed of my internet connection while using a VPN?"
#question = "Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?"
#question = "How can I be a good geologist?"
#question = "When do you use シ instead of し?"
#question = "What are the laws to change your status from a student visa to a green card in the US, how do they compare to the immigration laws in Canada?"
#question = "What are the laws to change your status from a student visa to a green card in the US? How do they compare to the immigration laws in Japan?"
#question = "What's causing someone to be jealous?"
#question = "How much is 30 kV in HP?"
#question = "I'm a 19-year-old. How can I improve my skills or what should I do to become an entrepreneur in the next few years?"
question = """
How many years Britain ruled India?
"""

def string_preprocessing(question):
    if question:
        text = expand_contractions(question, CONTRACTION_MAP)
        text = text.lower()
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
    else:
        text = ""

    return text

a = string_preprocessing(question)
print(a)
```
