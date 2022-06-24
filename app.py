from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import pickle
from tensorflow.keras.layers import *
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from gevent.pywsgi import WSGIServer
from waitress import serve

app = Flask(__name__)
CORS(app)
class Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, first_embedding, second_embedding):
        return tf.math.abs(first_embedding - second_embedding)
with open('tokenizer3.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('emb.pickle', 'rb') as f:
    emb_layer = pickle.load(f)
model = tf.keras.models.load_model('fifthSiamese.h5', custom_objects={'Dist': Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy()})
@app.route('/', methods=['POST', 'GET'])
def home():
    
    if request.method == 'POST':
        
        stuff = request.get_json()
        # stuff = {
        #     'search': ['eat fish'],
        #     'titles': ["Honey Butter Old Bay Salmon Recipe! ", '7 Ways to Detox and Cleanse Your Liver Naturally', 'Lower Your Cancer Risk', 'Are you ready?  Be happy and happier!', 'DRINK 1 CUP DAILY to Normalize High Blood Pressure', "It's never too late to become an entrepreneur!", 'Weak people, strong people, intelligent people', 'Celebrating Easter', 'How long have you been holding on to your glass of water?  ', 'Never Lost', 'How to Stop BLOATING Fast / Learn the 5 Causes - Dr. Berg', 'Do These Stretching Exercises Every Morning!', 'Brighter Days - Blessing Offor', 'Nothing is better than You!', 'River flows in you ', '8 ANTI-INFLAMMATORY DRINKS | to enjoy for health & wellness', 'Beautiful flowers! Wonderful creation!', 'Wanna play?  Chess?', 'Brave', 'What makes a good life? Lessons from the longest study on happiness | Robert Waldinger', 'Lessons from Winter', 'Raindrops Keep Falling On My Head', 'Tired of negativity and bad news? You are welcome to join in to breathe some fresh air!', '1 Vitamin Like Chemical to Heal Neuropathy & Nerve Damage | Dr. Alan Mandell, DC', 'Amazing benefits of garlic!', '8 Quick and Healthy Breakfast Recipes', 'Find something to be grateful for!', 'Happy new year to you and your love ones!  ', "It's better to be annoyingly positive....", 'The 7 Healthiest Foods You Should Eat - Dr. Berg', 'Benefits of Intermittent Fasting + How to Do It', 'Today is a gift!']
        # }
        # print(tokenizer.texts_to_sequences(["I play with toys"]))
        # search = tokenizer.texts_to_sequences(stuff['search'])
        # titles = tokenizer.texts_to_sequences(stuff['titles'])
        searcharr = []
        for i in range(len(stuff['titles'])):
            searcharr.append(stuff['search'][0])
        
        # x = pad_sequences(np.array(searcharr), maxlen=58, padding='post')
        # y = pad_sequences(np.array(titles), maxlen=58, padding='post')
        # scores = score(model, (x, y))
        # print(scores)
        matchingSentences = process_sentences(emb_layer, searcharr, stuff['titles'])
        output = {
            "info": matchingSentences
        }
        # for i in range(len(stuff['titles'])):
        #     output[stuff['titles'][i]] = float(scores[i][0])
        print(output)
        return jsonify(output)
    else:
        return render_template('index.html')

def score(model, sentences):
    return model.predict(sentences)

def process_sentences(layer, search, arr):
    # sentences = remove_unnecessary(arr)
    # print(search)
    # search_words = remove_unnecessary(search)
    
    # sentences = tokenizer.texts_to_sequences(sentences)
    # search_words = tokenizer.texts_to_sequences(search_words)
    # output = []
    # for j, sentence in enumerate(sentences):
    #     pas = False
    #     if sentence == []:
    #         continue
    #     for i in range(len(sentence)):
    #         for k in range(len(search_words)):
    #             for n in range(len(search_words[k])):
    #                 # print(tokenizer.sequences_to_texts([[sentence[i]]]))
    #                 # print(tokenizer.sequences_to_texts([[search_words[k][n]]]))
    #                 # print(siameseNet.predict((np.array([[sentence[i]]]), np.array([[search_words[k][n]]])))[0][0])
    #                 if float(model.predict((np.array([[sentence[i]]]), np.array([[search_words[k][n]]])))[0][0]) >= 0.95 or sentence[i] == search_words[k][n]:
    #                     pas = True
    #         # print(new_arr)
    #     if pas:
    #         output.append(arr[j])
    # return output
    searches = np.array(emb_layer(np.array(pad_sequences(tokenizer.texts_to_sequences(search), maxlen=100, padding='post', truncating='post')).astype(float)))
    titles = np.array(pad_sequences(tokenizer.texts_to_sequences(arr), maxlen=100, padding='post', truncating='post'))

    titles = np.array(emb_layer(titles))
    
    
    
    scores = tf.math.abs(tf.keras.losses.cosine_similarity(np.array(searches).mean(axis=1), np.array(titles).mean(axis=1)))
    output = []
    for i in range(len(scores)):
        if float(scores[i]) >= 0.60:
            output.append(arr[i])
    return output

def remove_unnecessary(arr):
    removal_words = ['a', 'the', 'at', 'for', 'is', 'in', 'was', 'could', 'are', 'were', 'that', 'there', 'this', 'an', '.', "on", "of", "as", 'it', 'by', 'to', 'and']
    output = []
    for sentence in arr:
        new_sentence = sentence.lower()
        curr_sentence = ""
        new_arr = []
        for i in range(len(new_sentence.split())):
            pas = True
            for k in range(len(removal_words)):
                if new_sentence.split()[i] == removal_words[k]:
                    pas = False
            if pas:
                new_arr.append(new_sentence.split()[i])
            # print(new_arr)
        for word in new_arr:
            curr_sentence += word + " "
        output.append(curr_sentence)
    return output
# gcloud builds submit --tag gcr.io/uw-cs400/siamese-container
# gcloud run deploy --image gcr.io/uw-cs400/siamese-container
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080, url_scheme='https')