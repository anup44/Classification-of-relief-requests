import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import json

from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda, Layer, concatenate, Concatenate, Reshape, Conv2D, Conv1D
from tensorflow.keras.models import Model, load_model, model_from_json
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn import datasets
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template
from pathlib import Path
nltk.download('wordnet')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
# graph = tf.get_default_graph()
# session  = tf.Session(graph)
# K.set_session(session)
# session.run(tf.global_variables_initializer())
# session.run(tf.tables_initializer())

FILE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = Path.home().joinpath('.requst_classifier')

with open(Path.joinpath(FILE_DIR, 'config.json'), 'r') as f:
    CONFIG_DICT = json.load(f)

nlp = spacy.load(MODEL_DIR.joinpath("en_core_web_lg"))
filter_POS = set(['NOUN', 'ADJ', 'VERB', 'PART'])
lemmatizer = WordNetLemmatizer()

def filter_query(query):
    query = query.lower()
    query = re.sub(r'[\t\n\r\f ]+', ' ', re.sub(r'\.', '. ', query))
    # print (query)
    doc = nlp(query)
    if len(doc) > 6:
        tokens = [t.text for t in doc if t.pos_ in filter_POS]
    else:
        tokens = [t.text for t in doc]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    filt_q = ' '.join(tokens)
    filt_q = re.sub(r'\b(n\'t|nt)\b', 'not', filt_q)
    filt_q = re.sub(r'\'ll\b', 'will', filt_q)
    return filt_q

categories = ['communication', 'essential_items', 'food', 'healthcare', 'others', 'travel']

category_counts = len(categories)
module_url = os.path.join(MODEL_DIR, CONFIG_DICT['models']['tf_sentence_encoder']['extracted_dir_name'])

session = tf.Session()
K.set_session(session)

embed = hub.Module(module_url)
embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)),signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(UniversalEmbedding,output_shape=(embed_size,))(input_text)
dense1 = Dense(1024, activation='relu')(embedding)
dense2 = Dense(128, activation='relu')(dense1)
pred = []
for c in categories:
    pred.append(Dense(1, activation='sigmoid')(dense2))
out = Concatenate(axis=-1)(pred)
model_use = Model(inputs=[input_text], outputs=out)


# nnlm_dim=128
# nnlm_input = Input(shape=(), dtype=tf.string)
# nnlm_embedding = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2", output_shape=[128], input_shape=[], dtype=tf.string, trainable=False)(nnlm_input)
# x = Dense(512, activation='relu')(nnlm_embedding)
# x = Dense(512, activation='relu')(x)
# out = Dense(len(categories), activation='sigmoid')(x)
# model_nnlm = Model(inputs=[nnlm_input], outputs=out)

# load json and create model
# json_file = open(os.path.join(FILE_DIR, 'models/model_comb_arch.h5'), 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})
# # load weights into new model
# loaded_model.load_weights(os.path.join(FILE_DIR, 'models/model_comb_test.h5'))
# print("Loaded model from disk")


input1 = Input(shape=(6), dtype=tf.float32)
input2 = Input(shape=(6), dtype=tf.float32)
reshaped1 = Reshape((6, 1, 1))(input1)
reshaped2 = Reshape((6, 1, 1))(input2)
concat = Concatenate(axis=2)([reshaped1, reshaped2])
conv = Conv2D(1, (1,2), padding='valid', use_bias=False)(concat)
reshaped = Reshape((6,))(conv)

model_comb = Model(inputs=[input1, input2], outputs=reshaped)

# init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

# session.run(init_op)
session.run(tf.global_variables_initializer())

model_use.load_weights(CONFIG_DICT['models']['tf_encoder_top_classifier']['download_path'])
model_nnlm = load_model(CONFIG_DICT['models']['nnlm_model']['download_path'], custom_objects={'KerasLayer': hub.KerasLayer})
# model_nnlm.load_weights(os.path.join(FILE_DIR, 'models/model_nnlm128_512_512.h5'))
model_comb.load_weights(CONFIG_DICT['models']['model_comb']['download_path'])

session.run(tf.tables_initializer())

models = [model_use, model_nnlm]
graph = tf.get_default_graph()
# model_nnlm.save(os.path.join(FILE_DIR, 'models/model_nnlm_weights_and_architecture_1.h5'))
# model_json = model_nnlm.to_json()
# with open(os.path.join(FILE_DIR, 'models', 'model_comb_arch.h5'), 'w') as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model_nnlm.save_weights(os.path.join(FILE_DIR, 'models', 'model_comb_test.h5'))
# print("Saved model to disk")

def get_scores(query_list):
    global graph
    filt_query = [filter_query(q) for q in query_list]
    print (filt_query)
    # filt_query = query_list
    with session.as_default():
        with graph.as_default():
            # query_arr = np.array(filt_query, dtype=object)[:, np.newaxis]
            query_arr1 = np.array(filt_query, dtype=object)[:, np.newaxis]
            query_arr2 = np.array(filt_query)
            predicts1 = model_use.predict(query_arr1)
            predicts2 = model_nnlm.predict(query_arr2)
            print (predicts1, predicts2)
            predicts = model_comb.predict([predicts1, predicts2])
    predicts[:, 4] = 0
    mask = np.ones((len(filt_query)), dtype=bool)
    mask[np.where(filt_query)] = False
    predicts[mask, :] = 0.0
    predicts[mask, 4] = 1.0
    return predicts

class_list = categories

def run_prediction(query_list):
    prob = get_scores(query_list)
    cutoff_score = 0.5
    # prob[:, 4] = 0
    # mask = np.ones((len(query_list)), dtype=bool)
    # mask[np.where(query_list)] = False
    # prob[mask, :] = 0.0
    # prob[mask, 4] = 1.0
    print ('**************************************************')
    print (prob)
    pred_results = []
    for i in range(prob.shape[0]):
        if len(query_list[i].split()) > 6:
            pred_labels = np.where(prob[i] > cutoff_score)[0]    
        else:
            pred_l = np.argmax(prob[i])
            pred_labels = [pred_l] if prob[i, pred_l] > cutoff_score else []
        pred_classes = [class_list[c] for c in pred_labels]
        if not pred_classes:
            pred_classes = ['others']
        pred_results.append(pred_classes)
    return pred_results

def run_prediction_single_label(query_list):
    prob = get_scores(query_list)
    prob[:, 4] = 0
    print (prob)
    pred_results = []
    for i in range(prob.shape[0]):
        pred_labels = [np.argmax(prob[i])]
        pred_classes = [class_list[c] for c in pred_labels]
        if not pred_classes:
            pred_classes = ['others']
        pred_results.append(pred_classes)
    return pred_results
