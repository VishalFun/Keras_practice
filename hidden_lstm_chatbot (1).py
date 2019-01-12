# -*- coding: utf-8 -*-

#importing necessary libraries

from keras.layers import LSTM,Embedding,Dense,Input,Dropout,TimeDistributed
from keras.models import Model
import numpy as np
import re
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import sys




dim = 500
epoch = 1200
# Reading and Extracting Word and their Vector
    
embeddings_index = {}
f = open('glove.6B.100d.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#Reading and Extracting Cornell Movie Dialogs and Creating Question and Answer

lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

questions = []
answers = []
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

  
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append('<BOS> '+clean_text(answer)+' <EOS>')
clean_answers[:1]

'''
removing words from senteces after given maxlen
''''
def tok(data,maxlen=20):
    new = []
    for i in data:
      tk = text_to_word_sequence(i)
      if len(tk)>=maxlen:  
        tk[maxlen-1:]=[]
        tk.append('eos')
      else:
        tk.append('eos')
      new.append(tk)
    return new
que = tok(clean_questions)
ans = tok(clean_answers)

#building vocabulary for model

def vocab(data):
  vocab = {}
  for i in data:
    for j in i:
      vocab[j]=vocab.get(j,0)+1
  return vocab

voc = vocab(que+ans)

int_to_word = {}
word_to_int = {}
vocab_size = len(voc)+1
for w,c in voc.items():
  word_to_int[w]=c
for w,i in word_to_int.items():
  int_to_word[i]=w


question_max = max([len(i)for i in que])
answer_max = max([len(i)for i in ans])


qd = np.zeros((len(que),question_max),dtype='float32')
ad = np.zeros((len(ans),answer_max),dtype='float32')
final_target_1 = np.zeros((len(ans),answer_max,vocab_size),dtype='float32')

for pos,i in enumerate(que):
  for pos1,j in enumerate(i):
    qd[pos,pos1]=word_to_int[j]

for pos,i in enumerate(ans):
  for pos1,j in enumerate(i):
    ad[pos,pos1]=word_to_int[j]


for pos,i in enumerate(ans):
  for pos1,j in enumerate(i):
    if pos1>0:
      final_target_1[pos,pos1-1,word_to_int[j]]=1
  

embedding_matrix = np.zeros((vocab_size,50))
for w,i in word_to_int.items():
  vec = embeddings_index.get(w)
  if vec is not None:
    embedding_matrix[i]
    


#model for chatbot
enc_inp = Input(shape=(None,))
dec_inp = Input(shape=(None,))

emb=Embedding(input_dim=vocab_size,output_dim=50,weights=[embedding_matrix],trainable=False)

lstm_cell = LSTM(dim,return_state=True)
lstm_bck = (LSTM(dim,return_sequences=True))
lstm_fun = LSTM(dim,return_sequences=True)
lstm_decoder = LSTM(dim,return_state=True,return_sequences=True)

dense = TimeDistributed(Dense(vocab_size,activation='softmax'))

inp_q_emb=emb(enc_inp)
inp_a_emb = emb(dec_inp)

#encoder model
inp_bck = lstm_bck(inp_q_emb)
encoder_lstm,state_h,state_c = lstm_cell(inp_bck)
encoder_states = [state_h,state_c]
#decoder model
decoder_lstm,_,_ = lstm_decoder(inp_a_emb,initial_state=encoder_states)
drp = Dropout((0.2))
drp1 = drp(decoder_lstm)
one = lstm_fun(decoder_lstm)
one1 = drp(one)
out = dense(one1)
main_model  = Model([enc_inp,dec_inp],out)
main_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
main_model.summary()

main_model.fit([qd,ad],final_target_1,epochs=epoch,batch_size=128,validation_split=0.02) #start training model upto given epochs

#encoder state model
enc_model = Model(enc_inp,encoder_states)
enc_model.summary()

#decoder model
dec_inp_h = Input(shape=(500,))
dec_inp_c = Input(shape=(500,))
dec_int_states = [dec_inp_h,dec_inp_c]

target,d_h,d_c = lstm_decoder(inp_a_emb,initial_state=dec_int_states)
states2 = [d_h,d_c]
final_out = dense(target)
decoder_model = Model([dec_inp]+dec_int_states,[final_out]+states2)
decoder_model.summary()


def chat(answer,question_h,question_c):
    flag = 0
    answer_1 = []
    i=0
    while flag != 1:
        prediction, prediction_h, prediction_c = decoder_model.predict([answer, question_h, question_c])
        token_arg = np.argmax(prediction[0, -1, :])
        got = int_to_word.get(token_arg)
        answer_1.append(got)
        if token_arg == word_to_int['eos'] or i > 20:
            flag = 1
        
        
        #answer[0, 0:-1] = answer[0, 1:]
        answer = np.zeros((1,1))
        answer[0,0] = token_arg
        question_h = prediction_h
        question_c = prediction_c
        i+=1
    print(" ".join(answer_1))

#Chatting with model
question = input("Enter message:")  
question = text_to_word_sequence(question)
adf = [word_to_int[i]for i in question]
adf = pad_sequences([adf],maxlen=question_max,padding='post',dtype='float32')
question_h, question_c = enc_model.predict(adf)
answer = np.zeros((1, 1))
answer[0,0] = word_to_int['bos']

chat(answer,question_h,question_c)




