import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from tensorflow import keras
from keras.preprocessing import sequence


reloaded_model = keras.models.load_model('./Models/nlp_model_A01_train_loss_weight.h5')
reloaded_model.summary()

reloaded_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

with open('./Models/nlp_model_A01_token.pickle', 'rb') as handle:
    token = pickle.load(handle)
# print(token.word_index)              

SentimentDict={0:'Video_Game_Category',1:'Community',2:'Food',3:'International',4:'Entertainment'}
def predict_article(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen = 500)
    predict_result = reloaded_model.predict_classes(pad_input_seq)
    print('Label:', np.argmax(reloaded_model.predict(pad_input_seq)))
    print('Category:', SentimentDict[predict_result[0]])

# predict_article('''
# In a shock announcement last week, Warner Bros said its 2021 slate of films - including Matrix 4, The Suicide Squad and Dune - would launch on HBO Max at the same time as cinemas in the US. It came after the studio had already revealed that the Wonder Woman sequel will stream at the same time as hitting the big screen on Christmas Day.
# While it means more fans will get to see the films during the COVID-19 pandemic, the "unprecedented" move has been seen by many as a huge blow to the struggling cinema industry.
# Gal Gadot is a woman in demand, back again as Wonder Woman. Pic: Warner Bros
# Gal Gadot's return as Wonder Woman will hit HBO Max as well as cinemas on Christmas Day in the US. Pic: Warner Bros
# Cinemas have been hit by a raft of delays to high-profile films in 2020, including the latest instalments in popular franchises such as Marvel and James Bond.
# One of Hollywood's biggest directors, Nolan is famous for films including Inception, Dunkirk, the Dark Knight Batman trilogy, and most recently the long-awaited Tenet earlier in 2020, and is a big advocate for experiencing films on the big screen.
# Responding to Warner Bros' decision, he said his first reaction was "disbelief".
# Speaking to Entertainment Tonight, Nolan said: "There's such controversy around it, because they didn't tell anyone.
# ''')