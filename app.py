import pickle

import flask
import numpy as np
import numpy as np
import pickle


from tensorflow import keras
from flask import Flask, render_template, url_for, request, jsonify
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from utils.utils import *
import logging
import os

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

assets_path = 'assets'
max_text_len = 140

app = Flask(__name__)


# ml assets

with open(os.path.join(assets_path, 'mlmodel.pkl'), 'rb') as model:
    ml_model = pickle.load(model)

with open(os.path.join(assets_path, 'le.pkl'), 'rb') as lencoder:
    label_encoder = pickle.load(lencoder)


# dl assets

# model
dlmodel = keras.models.load_model(os.path.join(assets_path, 'dlmodel.h5'))

# tokenizer
with open(os.path.join(assets_path, 'tokenizer.pkl'), 'rb') as text_tokenizer:
    x_tokenizer = pickle.load(text_tokenizer)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':

        if request.form['action'] == "نموذج تعلم الآله":
            logging.info(request)
            text = request.form['Text']
            logging.info(request)

            logging.info("reesponse text type:")
            logging.info(type(text))

            text = preprocessing(text)
            pred = ml_model.predict([text])
            res = label_encoder.inverse_transform(pred)

            return render_template('pred.html', data=res[0])
        elif request.form['action'] == "نموذج التعلم العميق":

            text = request.form['Text']
            text = preprocessing(text)
            text = x_tokenizer.texts_to_sequences([text])
            text = pad_sequences(text,  maxlen=max_text_len, padding='post')
            pred = dlmodel.predict(text)
            top = np.argmax(pred, axis=-1)
            k = 4
            top_k_index = np.argsort(pred, axis=-1)[:, ::-1][:, :k]
            top_k_props = pred[np.arange(pred.shape[0])[:, None], top_k_index]
            top_k_code = [label_encoder.inverse_transform(
                top_k_index[i]).tolist() for i in range(top_k_index.shape[0])]

            res = label_encoder.inverse_transform(top)

            return render_template('preddl.html', data=res[0], codes=top_k_code, probs=top_k_props)


if __name__ == '__main__':
    app.run(port=8888, debug=True)
