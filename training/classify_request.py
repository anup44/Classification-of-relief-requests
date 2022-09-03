import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from tensorflow import keras
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from keras import backend as K
from flask import Flask, request, render_template

graph = tf.get_default_graph()

session  = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())

module_url = "../tf_sent_encoder_2"
categories = ['abuse','communication','food','fuel','health','travel']
embed = hub.Module(module_url)
embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value

category_counts = len(categories)

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)),signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(UniversalEmbedding,output_shape=(embed_size,))(input_text)
dense1 = Dense(1024, activation='relu')(embedding)
dense2 = Dense(128, activation='relu')(dense1)
pred = Dense(category_counts, activation='sigmoid')(dense2)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())

model.load_weights('./model4.h5')

categories = ['abuse','communication','food','fuel','health','travel']

def predict_query_class(query):
    global graph
    with graph.as_default():
        new_text = np.array([query, ''], dtype=object).reshape((-1, 1))
        # print (new_text.shape)
        predicts = model.predict(new_text)
        # print (predicts)
        predicts = predicts[0]
        predicted_classes = [categories[i] for i in range(len(predicts)) if predicts[i] >= 0.5]
        return predicted_classes

app = Flask(__name__)

@app.route('/')
def query_form():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def run_classifier():
    multi_label_flag = 'off'

    if request.method == 'GET':
        query_text = request.args.get('query', default='no query specified', type=str)

    else:
        query_text = request.form['query']
        multi_label_flag = request.form['multi_label']
        # print (multi_label_flag)
    # if multi_label_flag == 'off':
    #     result = classify_request(query_text)
    # else:
    #     result = classify_request_multi_label(query_text)
    result = predict_query_class(query_text)
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
    # if multi_label_flag == 'false':
    #     result = classify_request(query_text)
    # else:
    #     result = classify_request_multi_label(query_text)
    # return result
    # print (type(query_text))
    result = predict_query_class(query_text)
    # out = [class_dict[r] for r in result]
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
    # print ('app started')
    # while True:
    #     query = input()
    #     print (classify_request(query))