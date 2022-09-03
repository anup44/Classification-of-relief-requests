import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import os
import pandas as pd
import re

from tensorflow import keras
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, Concatenate
from keras.models import Model
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.optimizers import Adam

import random
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn import datasets
import spacy
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template

# graph = tf.get_default_graph()
# session  = tf.Session(graph)
# K.set_session(session)
# session.run(tf.global_variables_initializer())
# session.run(tf.tables_initializer())

df = pd.DataFrame(columns=['label','text'])
trainset = datasets.load_files(container_path = 'aabhas/dataset_small_less_categories',encoding = 'UTF-8')

def get_dataframe(lines):
#     lines = filename.splitlines()
    data = []
    for i in range(0, len(lines)):
        label = lines[i][0]
    #     label = label.split(",")[0]
        text = ' '.join(lines[i][1:])
        text = re.sub(',','', text)
        data.append([label, text])
    df = pd.DataFrame(data, columns=['label', 'text'])
    df.label = df.label.astype('category')
    return df

for k in range(len(trainset.target)):
    labelled_sent= trainset.data[k].strip().split('\r\n')
    Label=[trainset.target[k]]*len(labelled_sent)
    labelled_emb = list(zip([s for s in Label], [s for s in labelled_sent]))
    df_train = get_dataframe(labelled_emb)
    df=pd.concat([df,df_train])
    df.reset_index(inplace=True, drop=True)


data=[]
length=len(df)

for i in range(length * 10):
    num1 = i % length
    num2 = num1
    while (num1 == num2):
        num2=random.randrange(length)
    data.append([df.label[num1],df.label[num2],df.text[num1] + ' ' + df.text[num2]])
    
for i in range(length):
    data.append([df['label'][i], df['label'][i], df['text'][i]])     
df2 = pd.DataFrame(data, columns=['label1','label2', 'text'])


nlp = spacy.load("en_core_web_lg")
filter_POS = set(['NOUN', 'NUM', 'ADJ', 'VERB', 'PART'])
lemmatizer = WordNetLemmatizer()

def filter_query(query):
    query = query.lower()
    query = re.sub(r'[\t\n\r\f ]+', ' ', re.sub(r'\.', '. ', query))
    # print (query)
    doc = nlp(query)
    tokens = [t.text for t in doc if t.pos_ in filter_POS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


category_counts = len(trainset.target_names)
module_url = "tf_sent_encoder_2" 

# graph = tf.Graph()
session = tf.Session()
K.set_session(session)

# with graph.as_default():
embed = hub.Module(module_url)
embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)),signature="default", as_dict=True)["default"]

# def sentence_encoder(input_text):
#     return embed(tf.squeeze(input_text))
categories = trainset.target_names

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(UniversalEmbedding,output_shape=(embed_size,))(input_text)
dense1 = Dense(1024, activation='relu')(embedding)
dense2 = Dense(128, activation='relu')(dense1)
pred = []
for c in categories:
    pred.append(Dense(1, activation='sigmoid')(dense2))
out = Concatenate()(pred)
model = Model(inputs=[input_text], outputs=out)

init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    
# graph.finalize()

# LEARNING_RATE = 0.001

# optimizer = Adam(lr=LEARNING_RATE)

# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

session.run(init_op)
model.load_weights('aabhas/model10.h5')
graph = tf.get_default_graph()

def get_scores(query_list):
    global graph
    filt_query = [filter_query(q) for q in query_list]
    print (filt_query)
    # filt_query = query_list
    with graph.as_default():
        query_arr = np.array(filt_query, dtype=object)[:, np.newaxis]
        predicts = model.predict(query_arr)
    return predicts

class_list = trainset.target_names

def run_prediction(query_list):
    prob = get_scores(query_list)
    prob[:, 3] = 0
    print (prob)
    pred_results = []
    for i in range(prob.shape[0]):
        if len(query_list[i].split()) > 6:
            pred_labels = np.where(prob[i] > 0.5)[0]    
        else:
            pred_labels = [np.argmax(prob[i])]
        pred_classes = [class_list[c] for c in pred_labels]
        if not pred_classes:
            pred_classes = ['others']
        pred_results.append(pred_classes)
    return pred_results

def run_prediction_single_label(query_list):
    prob = get_scores(query_list)
    prob[:, 3] = 0
    print (prob)
    pred_results = []
    for i in range(prob.shape[0]):
        pred_labels = [np.argmax(prob[i])]
        pred_classes = [class_list[c] for c in pred_labels]
        if not pred_classes:
            pred_classes = ['others']
        pred_results.append(pred_classes)
    return pred_results

# text = 'no news information given on ration supplies in nearby area'
# print (run_prediction([text, ''])[0])


app = Flask(__name__)

@app.route('/')
def query_form():
    return render_template('index.html', checked='on')

@app.route('/', methods=['GET', 'POST'])
def run_classifier():
    multi_label_flag = 'off'

    if request.method == 'GET':
        query_text = request.args.get('query', default='no query specified', type=str)

    else:
        query_text = request.form['query']
        multi_label_flag = request.form['multi_label']
        # print (multi_label_flag)
    if multi_label_flag == 'off':
        result = run_prediction_single_label([query_text, ''])[0]
    else:
        result = run_prediction([query_text, ''])[0]
    # result = run_prediction([query_text, ''])[0]
    print (result)
    # return result
    return render_template('index.html', text_query=query_text, out_class=result, checked=multi_label_flag)


@app.route('/get_category', methods=['GET', 'POST'])
def ajax_response():
    multi_label_flag = 'false'
    print ('lol')
    if request.method == 'GET':
        query_text = request.args.get('query', default='no query specified', type=str)

    else:
        print (request.form)
        query_text = request.form['query']
        multi_label_flag = request.form['multi_label']
        print (multi_label_flag)
    if multi_label_flag == 'false':
        result = run_prediction_single_label([query_text, ''])[0]
    else:
        result = run_prediction([query_text, ''])[0]    
    # return result
    # print (type(query_text))
    # result = run_prediction([query_text, ''])[0]
    # out = [class_dict[r] for r in result]
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
    # print ('app started')
    # while True:
    #     query = input()
    #     print (classify_request(query))

