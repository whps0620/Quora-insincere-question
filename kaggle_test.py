import numpy as np 
import pandas as pd
import tensorflow as tf
import jieba
from collections import Counter
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, RNN
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *

train= pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
train_df, test_df = train_test_split(train, test_size = 0.2, random_state = 42)
train_data, val_data = train_test_split(train_df, test_size = 0.2, random_state = 42)
print('real train: train_df', train_df.shape)
print('real test: test_df', test_df.shape)
print('train of train: train_data', train_data.shape)
print('validation : val_data', val_data.shape)

# take 1000 piece of data to build model

# ------------------basic EDA ------------------------
# now let us check in the number of Percentage
Count_Normal_transacation = len(train_data[train_data["target"]==0])
# normal transaction are repersented by 0
Count_insincere_transacation = len(train_data[train_data["target"]==1]) 
# fraud by 1
Percentage_of_Normal_transacation = Count_Normal_transacation/(Count_Normal_transacation+Count_insincere_transacation)
print("percentage of normal transacation is",Percentage_of_Normal_transacation*100)

Percentage_of_insincere_transacation= Count_insincere_transacation/(Count_Normal_transacation+Count_insincere_transacation)
print("percentage of fraud transacation",Percentage_of_insincere_transacation*100)

insincere_indices= np.array(train_data[train_data.target==1].index)
normal_indices = np.array(train_data[train_data.target==0].index)
#now let us a define a function for make undersample data with different proportion
#different proportion means with different proportion of normal classes of data
#times denote the normal data = times*fraud data
Normal_indices_undersample = np.array(np.random.choice(normal_indices,(Count_insincere_transacation),replace=False))
print(len(Normal_indices_undersample))
undersample_data= np.concatenate([insincere_indices,Normal_indices_undersample])

undersample_data = train_data.iloc[undersample_data,:]
#print(undersample_data)
print(len(undersample_data))
print("the normal transacation proportion is :",len(undersample_data[undersample_data.target==0])/len(undersample_data))
print("the fraud transacation proportion is :",len(undersample_data[undersample_data.target==1])/len(undersample_data))
print("total number of record in resampled data is:",len(undersample_data))
Undersample_data = undersample_data(normal_indices,insincere_indices,1)


# -----------target distribution-------------------
labels = (np.array(cnt_srs.index))
sizes = (np.array((cnt_srs / cnt_srs.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Target distribution',
    font=dict(size=18),
    width=400,
    height=400,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename="usertype")


# train_data = train_data[:3000]
# val_data = val_data[:700]
# test_df = test_df[:300]

# ------------word frequency----------------------
### ------------1-gram-----------------
from plotly import tools
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
import plotly as py
stopwords = set(STOPWORDS)
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stopwords = stopwords.union(more_stopwords)

from collections import defaultdict
train1_df = train_df[train_df["target"]==1]
train0_df = train_df[train_df["target"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace


# ---------------前置作業----------------------
## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(5), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(5), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=1000, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.offline.iplot(fig, filename='word-plots')

###----------------bigram------------------------
freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(5), 'orange')


freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(5), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["Frequent bigrams of sincere questions", 
                                          "Frequent bigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
py.offline.iplot(fig, filename='word-plots') 

###---------------trigram---------------------
freq_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(5), 'green')
freq_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent,3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(5), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,
                          subplot_titles=["Frequent trigrams of sincere questions", 
                                          "Frequent trigrams of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=400, width=1000, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")
py.offline.iplot(fig, filename='word-plots')



# ------------------preprocess------------------------
from gensim.models import Word2Vec
print(model.wv['machine_learning'])
import spacy
import logging
import time
from gensim.corpora import WikiCorpus
import spacy.cli
nlp = spacy.load('en_core_web_md', disable=["ner", "parser"])
model_words = set(model.wv.index2word)
from gensim.models import KeyedVectors

# spacy.cli.download("en_core_web_md")
EMBEDDING_FILE = '/Users/windy/Desktop/kaggle-1/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_data["question_text"].fillna("_na_").values
val_X = val_data["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_data['target'].values
val_y = val_data['target'].values

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix[i] = embedding_vector

def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
y_test = model.predict(test_X)
y_test = y_test.reshape((-1, 1))

kaggle = test_df.copy()
pred_test_y = (y_test>0.34506).astype(int)
kaggle['prediction'] = pred_test_y
original_test_y = test_df['target']

##-------undersampling----------------
TODO:
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
rus = RandomUnderSampler(sampling_strategy='majority')
train_X,train_y = rus.fit_resample(train_X,train_y) 

##-----------oversampling---------------
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(sampling_strategy='minority')
train_X,train_y = ros.fit_resample(train_X,train_y) 

##----------------logistic regression----------
TODO:
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_X, train_y)
y_pred = classifier.predict(test_X)
y_pred_proba = classifier.predict_proba(test_X)
# -------------------GRU------------------------------
# inp = Input(shape=(maxlen,))
# x = Embedding(max_features, embed_size)(inp)
# x = Bidirectional(GRU(64, return_sequences=True))(x)
# x = GlobalMaxPool1D()(x)
# x = Dense(16, activation="relu")(x)
# x = Dropout(0.1)(x)
# x = Dense(1, activation="sigmoid")(x)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())

# -----------------------------先存檔---------------------------------
test_df.to_csv('kaggle_test.csv')
# ------------------------------읽기----------------------------------
# kaggle = pd.read_csv('kaggle_test.csv')
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
y_test = model.predict(test_X)
y_test = y_test.reshape((-1, 1))
kaggle = test_df.copy()
pred_test_y = (y_test>0.34506).astype(int)
kaggle['prediction'] = pred_test_y

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score,confusion_matrix
accuracy = accuracy_score(original_test_y, y_pred)
f1 = f1_score(original_test_y, y_pred)
precision = precision_score(original_test_y, y_pred)
recall = recall_score(original_test_y, y_pred)
cm = confusion_matrix(original_test_y, y_pred)
print('accuracy:', accuracy)
print('F1 score:', f1)
print('precision score:',precision)
print('recall score:', recall)
print('confusion matrix:', cm)

tp = []
tn = []
fp = []
fn = []
for i in range(len(kaggle)):
    if kaggle['target'].iloc[i]== 1 and kaggle['prediction'].iloc[i] == 1:
        tp.append(i)
    if kaggle['target'].iloc[i]== 0 and kaggle['prediction'].iloc[i] == 0:
        tn.append(i)
    if kaggle['target'].iloc[i] ==1  and kaggle['prediction'].iloc[i] == 0:
        fp.append(i)
    if kaggle['target'].iloc[i] ==0  and kaggle['prediction'].iloc[i] == 1:
        fn.append(i)
print(len(tn))
print(len(fp))
print(len(fn))
print(len(tp))




# -------------confusion metrix--------------------------

# 確認前面的test的y是誰
# y_pred：預測值 （0 or 1）
# y_test：實際值（真實的0 or 1）
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# y_pred : 預測值 (0/1)
# 
# ------------------Decision Tree------------------------------------


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(train_X, train_y)

y_pred = classifier.predict(test_X)

# -------------------SVM----------------------------------------------

# -------------------random forest------------------------------------