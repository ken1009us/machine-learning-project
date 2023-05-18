from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model

import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle


def read_file():
    path = './All_data_A01.csv'
    df = pd.read_csv(path, encoding='utf-8').astype("str")
    # print(df)
#     print(df[['content','category']].head(5))
    return df
# read_file()

def dataframe():
    df = read_file()
    df = df[['content','category']]
    # df_num = df.replace({'Video_Game_Category':0, 
    #                      'Fashion':1, 
    #                      'Politics':2, 
    #                      'Education':3, 
    #                      'Science':4,
    #                      'Music':5
    #                       ......
    #                       etc.})
    
    # df_num = df_num[df_num['category'].isin([0, 1, 2, 3, 4, 5, etc.])]
    
    return df, df_num

def token():
    all_data_list = []
    all_contents = []
#     all_categories = []
    df, df_num = dataframe()
    all_data_list = df_num.values.tolist()
    random.shuffle(all_data_list)
    for data in all_data_list:
        data[0], data[1]
        all_contents.append(data[0])
        #all_categories.append(data[1])

    token = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=None, split=" ", char_level=False)
    token.fit_on_texts(all_contents)
    # print(token.document_count)
    # print(token.word_index)

    with open('./Models_A01/nlp_model_A01_token.pickle', 'wb') as handle:
        pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return token


config = tf.ConfigProto()
#config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
##config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

G = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.42
session = tf.Session(config=config)

token()
token = token()
df_select = []
data_list = []
df, df_num = dataframe()

df_only_cate_num = df_num[['category']].drop_duplicates(subset = 'category')
cate_list = df_only_cate_num.values.tolist()

# df_cateEng = df_only_cate_num.replace({0:'Video_Game_Category', 
#                                        1:'Fashion', 
#                                        2:'Politics', 
#                                        3:'Education', 
#                                        4:'Science',
#                                        5:'Music',
#                                        ...
#                                        etc.})
cateEng_list = df_cateEng.values.tolist()

log_name = 'log_file_A01'
logpath = './' + log_name + '.log'
if os.path.isfile(logpath):
    os.remove(logpath)

for cate in cate_list:
    for cate_2 in cate_list:
        if cate != cate_2:
            df_select = df_num[df_num['category'].isin([cate[0], cate_2[0]])]
            data_list = df_select.values.tolist()
            random.shuffle(data_list)
            
            loadtrainlst, temp = train_test_split(data_list, test_size=0.5, random_state=42)
            loadvaildlst, loadtestlst = train_test_split(temp, test_size=0.5, random_state=42)
            # for data in data_list:
            #     print(data[1])
            
            x_loadtrain = []
            y_loadtrain = []
            for train_data in loadtrainlst:
                train_data[0], train_data[1]
                x_loadtrain.append(train_data[0])
                y_loadtrain.append(train_data[1])

            x_loadvalid = []
            y_loadvalid = []
            for valid_data in loadvaildlst:
                valid_data[0], valid_data[1]
                x_loadvalid.append(valid_data[0])
                y_loadvalid.append(valid_data[1])
    
            x_loadtest = []
            y_loadtest = []
            for test_data in loadtestlst:
                test_data[0], test_data[1]
                x_loadtest.append(test_data[0])
                y_loadtest.append(test_data[1])

            x_train_seq = token.texts_to_sequences(x_loadtrain)
            x_valid_seq = token.texts_to_sequences(x_loadvalid)
            x_test_seq = token.texts_to_sequences(x_loadtest)

            x_train = sequence.pad_sequences(x_train_seq, maxlen = 500)
            x_valid = sequence.pad_sequences(x_valid_seq, maxlen = 500)
            x_test = sequence.pad_sequences(x_test_seq, maxlen = 500)

            y_loadtrain = np.array(y_loadtrain)
            y_loadvalid = np.array(y_loadvalid)
            y_loadtest = np.array(y_loadtest)

            y_tr_OneHot = np_utils.to_categorical(y_loadtrain, 18)
            y_va_OneHot = np_utils.to_categorical(y_loadvalid, 18)
            y_te_OneHot = np_utils.to_categorical(y_loadtest, 18)
    
            x_train = np.array(x_train)
            x_valid = np.array(x_valid)
            x_test = np.array(x_test)
            
            model_name = str(cate[0])+'_'+str(cate_2[0])+'_'+'nlp_model_A01'
            model = Sequential()
            model.add(Embedding(output_dim=32,
                                input_dim=5000,
                                input_length=500))
            # model.add(Dropout(0.35))
            model.add(GRU(32))
            model.add(Dense(units=256, activation='relu'))
            model.add(Dropout(0.35))
            model.add(Dense(units=18,activation='sigmoid'))
            #model.summary()
            model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])
    
            model_train_loss_fullmode = ModelCheckpoint('./Models_A01/%s_train_loss_weight.h5' % model_name,
                                                         monitor='loss',
                                                         save_best_only=True,
                                                         mode='min')
    
            model_val_loss_fullmode = ModelCheckpoint('./Models_A01/%s_val_loss_weight.h5' % model_name,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min')
            
            model_early_stop = EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=10, 
                                             verbose=0, 
                                             mode='min', 
                                             baseline=None, 
                                             restore_best_weights=False)
    
            yaml_string = model.to_yaml()
            yaml_filepath = './Models_A01/' + model_name + '_' + '.yaml'
            with open(yaml_filepath, 'w') as outfile:
                outfile.write(yaml_string)
            
            labelmessage = 'Main:{0}({1})......Other:{2}({3})' .format(cateEng_list[cate[0]][0], cate[0], 
                                                                       cateEng_list[cate_2[0]][0], cate_2[0])
            print(labelmessage)
            
            train_history=model.fit(x_train, 
                                    y_tr_OneHot, 
                                    validation_data = (x_valid, y_va_OneHot),
                                    batch_size=100,
                                    epochs=10, 
                                    verbose=2,
                                    callbacks=[model_train_loss_fullmode, model_val_loss_fullmode, model_early_stop])
            
            valid_scores = model.evaluate(x_valid, y_va_OneHot, verbose=1)
            valid_score = round(valid_scores[1] * 100, 2)
            print('dataset [model:train] length:', len(y_va_OneHot))
            print('dataset [model:train] score:' ,valid_score)
#             print('-----------------------------------------------------------')
            
            test_scores = model.evaluate(x_test, y_te_OneHot, verbose=1)
            test_score = round(test_scores[1] * 100, 2)
            print('dataset [model:test] length:', len(y_te_OneHot))
            print('dataset [model:test] score:' ,test_score)
            print('-----------------------------------------------------------')
                
            wf = open(logpath, 'a')
            wf.write('Main:{0}({1})......Other:{2}({3})\n' .format(cateEng_list[cate[0]][0], cate[0], 
                                                                   cateEng_list[cate_2[0]][0], cate_2[0]))
            wf.write('validation dataset [model:loss] score :%s\n' % valid_score) 
            wf.write('test dataset [model:loss] score :%s\n' % test_score) 
            wf.write('--------------------------------------------------------\n')
            wf.close()
            
#             minscores = 80
#             if int(valid_score) < minscores or int(test_score) < minscores:
#                 train_filepath = './Models_A01/%s_train_loss_weight.h5' % model_name
#                 valid_filepath = './Models_A01/%s_val_loss_weight.h5' % model_name
                
#                 cmd_train = 'mv %s ./Models_A01/nogood_models/' % train_filepath
#                 os.system(cmd_train)
#                 cmd_valid = 'mv %s ./Models_A01/nogood_models/' % valid_filepath
#                 os.system(cmd_valid)  
#                 cmd_yaml = 'mv %s ./Models_A01/nogood_models/' % yaml_filepath
#                 os.system(cmd_yaml)

