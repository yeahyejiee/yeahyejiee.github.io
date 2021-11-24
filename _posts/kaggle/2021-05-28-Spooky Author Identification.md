---

author: Hone Ye ji
categories: 
 - Kaggle
 - ml
tags: 
 - kaggle
toc: true
toc_label: "My Table of Contents"
toc_icon: "cog"
---
kaggle의 Spooky Author Identification를 연습해보겠습니다.


# Spooky Author Identification
#### 공포이야기에 있는 문장의 단어를 분석하여 작가를 예측
#### 제출: id + 3명의 작가에 대한 확률 => 3개의 클래스로 텍스트 분류


```python
import pandas as pd
import numpy as np
```

### 1. 데이터 불러오기


```python
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id26305</td>
      <td>This process, however, afforded me no means of...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id17569</td>
      <td>It never once occurred to me that the fumbling...</td>
      <td>HPL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id11008</td>
      <td>In his left hand was a gold snuff box, from wh...</td>
      <td>EAP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27763</td>
      <td>How lovely is spring As we looked from Windsor...</td>
      <td>MWS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id12958</td>
      <td>Finding nothing else, not even gold, the Super...</td>
      <td>HPL</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (19579, 3)




```python
test.shape
```




    (8392, 2)




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id02310</td>
      <td>Still, as I urged our leaving Ireland with suc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id24541</td>
      <td>If a fire wanted fanning, it could readily be ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id00134</td>
      <td>And when they had broken down the frail door t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27757</td>
      <td>While I was thinking how I should possibly man...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id04081</td>
      <td>I am not sure to what limit his knowledge may ...</td>
    </tr>
  </tbody>
</table>
</div>



### 2.시각화


```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud
```

- 작가에 해당하는 단어


```python
train.author.value_counts().plot(kind="bar")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x219fb0a59a0>




![output_11_1](https://user-images.githubusercontent.com/45659433/119966780-9cdc5200-bfe6-11eb-8c7a-5f4892d78f69.png)


- 문장의 길이 알아보기


```python
data_length=train.text.apply(len)
data_length.head()
```




    0    231
    1     71
    2    200
    3    206
    4    174
    Name: text, dtype: int64




```python
plt.figure(figsize=(12,5))
plt.hist(data_length, bins= 20, range=[0,500], color="r", alpha=0.3)
plt.show()
```


![output_14_0](https://user-images.githubusercontent.com/45659433/119966783-9d74e880-bfe6-11eb-93f9-ba9491bf8c2e.png)


- 한문장에 대략 몇개의 단어가 들어가 있는지


```python
data_split_length= train.text.apply(lambda x:len(x.split(" ")))
data_split_length.head()
```




    0    41
    1    14
    2    36
    3    34
    4    27
    Name: text, dtype: int64




```python
plt.figure(figsize=(12,5))
plt.hist(data_split_length, bins=10, range=[0,100], color='b', alpha=0.5)
plt.show()
```


![output_17_0](https://user-images.githubusercontent.com/45659433/119966785-9e0d7f00-bfe6-11eb-8417-754c6104b885.png)


- 워드클라우드: 전체 텍스트에서 많이 사용되는 단어들


```python
cloud= WordCloud(width=400, height=200).generate(" ".join(train.text))
plt.figure(figsize=(12,5))
plt.imshow(cloud)
plt.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![output_19_1](https://user-images.githubusercontent.com/45659433/119966786-9e0d7f00-bfe6-11eb-986e-f1efc1c71e4e.png)


- 워드클라우드 저자별로 많이 사용되는 단어들

HPL


```python
cloud= WordCloud(width=400, height=200).generate(" ".join(train[train['author']=='HPL']['text']))
plt.figure(figsize=(12,5))
plt.imshow(cloud)
plt.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![output_22_1](https://user-images.githubusercontent.com/45659433/119966789-9ea61580-bfe6-11eb-9bb0-40f7b5ee9b2a.png)


MWS


```python
cloud= WordCloud(width=400, height=200).generate(" ".join(train[train['author']=='MWS']['text']))
plt.figure(figsize=(12,5))
plt.imshow(cloud)
plt.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![output_24_1](https://user-images.githubusercontent.com/45659433/119966790-9f3eac00-bfe6-11eb-9f76-b8e1401e5c9a.png)


EAP


```python
cloud= WordCloud(width=400, height=200).generate(" ".join(train[train['author']=='EAP']['text']))
plt.figure(figsize=(12,5))
plt.imshow(cloud)
plt.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![output_26_1](https://user-images.githubusercontent.com/45659433/119966792-9f3eac00-bfe6-11eb-95cc-213ed07562e0.png)


### 2. 데이터 전처리

##### 작가의 이름을 0,1,2로 변환


```python
from sklearn import preprocessing
from keras.preprocessing import  sequence, text
```


```python
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author)
y[:10]
```




    array([0, 1, 0, 2, 1, 2, 0, 0, 0, 2])



### 3. 데이터셋 나누기


```python
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train.text.values, y, stratify=y, random_state=42, test_size=0.3, shuffle=True)
```


```python
print( x_train.shape)
print( x_valid.shape)

print( y_train.shape)
print( y_valid.shape)
```

    (13705,)
    (5874,)
    (13705,)
    (5874,)
    

#### 원핫인코딩


```python
from keras.utils import np_utils

ytrain_enc = np_utils.to_categorical(y_train) 
yvalid_enc = np_utils.to_categorical(y_valid)
```

### 4. keras로 모델만들기

- 5000개의 단어 사용
- 최대 길이 60
- padding을 통해 길이 맞추기
- texts_to_sequences() 메서드를 이용해서 이러한 단어들을 시퀀스의 형태로 변환(word_index를 통해 텍스트 단어의 순서를 나열한 것을 각 문장에 맞게 변환) 


```python
from keras.preprocessing.text import Tokenizer

num_words=5000
max_len=50
emb_size=64

token= Tokenizer(num_words=num_words)  #, oov_token은 토큰화 되지 않은 단어에 대해 특수한 값으로 변환
token.fit_on_texts(list(x_train) + list(x_valid))
word_index = token.word_index
print(word_index)

xtrain_seq = token.texts_to_sequences(x_train)
xvalid_seq = token.texts_to_sequences(x_valid)
test_seq = token.texts_to_sequences(test.text.values)

# zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
test_pad = sequence.pad_sequences(test_seq , maxlen=max_len)
```

    {'the': 1, 'of': 2, 'and': 3, 'to': 4, 'a': 5, 'i': 6, 'in': 7, 'was': 8, 'that': 9, 'my': 10, 'it': 11, 'had': 12, 'he': 13, 'with': 14, 'his': 15, 'as': 16, 'for': 17, 'which': 18, 'but': 19, 'not': 20, 'at': 21, 'me': 22, 'from': 23, 'by': 24, 'is': 25, 'this': 26, 'on': 27, 'be': 28, 'her': 29, 'were': 30, 'have': 31, 'all': 32, 'you': 33, 'we': 34, 'or': 35, 'no': 36, 'an': 37, 'one': 38, 'so': 39, 'him': 40, 'when': 41, 'been': 42, 'they': 43, 'upon': 44, 'there': 45, 'could': 46, 'she': 47, 'its': 48, 'would': 49, 'more': 50, 'now': 51, 'their': 52, 'what': 53, 'some': 54, 'our': 55, 'are': 56, 'into': 57, 'than': 58, 'will': 59, 'very': 60, 'who': 61, 'if': 62, 'them': 63, 'only': 64, 'then': 65, 'up': 66, 'these': 67, 'before': 68, 'about': 69, 'any': 70, 'time': 71, 'man': 72, 'yet': 73, 'out': 74, 'said': 75, 'even': 76, 'did': 77, 'your': 78, 'might': 79, 'after': 80, 'old': 81, 'like': 82, 'first': 83, 'us': 84, 'must': 85, 'most': 86, 'through': 87, 'over': 88, 'never': 89, 'made': 90, 'life': 91, 'night': 92, 'found': 93, 'such': 94, 'other': 95, 'should': 96, 'do': 97, 'seemed': 98, 'eyes': 99, 'every': 100, 'little': 101, 'while': 102, 'those': 103, 'still': 104, 'day': 105, 'myself': 106, 'great': 107, 'long': 108, 'saw': 109, 'has': 110, 'where': 111, 'own': 112, 'many': 113, 'well': 114, 'again': 115, 'came': 116, 'much': 117, 'down': 118, 'may': 119, 'thought': 120, ...}
    


```python
xtrain_seq[:2]
```




    [[33, 116, 1, 987, 2, 10, 331],
     [33, 10, 2268, 22, 53, 255, 123, 6, 3278, 23, 78, 445, 807, 61, 22, 166]]




```python
xtrain_pad[:2]
```




    array([[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   33,
             116,    1,  987,    2,   10,  331],
           [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
               0,   33,   10, 2268,   22,   53,  255,  123,    6, 3278,   23,
              78,  445,  807,   61,   22,  166]])



## 5.모델

## 모델1:  Bidirectional LSTM 


```python
from keras.layers import LSTM, Input, Dropout, Bidirectional, GlobalMaxPool1D, Embedding, Dense
from keras.models import Model
```


```python
def model_1():
    inp= Input(shape=(max_len,))
    layer=Embedding(num_words, emb_size)(inp)
    layer=Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer=GlobalMaxPool1D()(layer)
    layer=Dropout(0.2)(layer)
    layer=Dense(16, activation ='relu')(layer)
    layer=Dropout(0.2)(layer)
    layer=Dense(3, activation ='softmax')(layer)
    model=Model(inputs=inp, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
```


```python
model_1=model_1()
model_1.summary()
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 50)]              0         
    _________________________________________________________________
    embedding (Embedding)        (None, 50, 64)            320000    
    _________________________________________________________________
    bidirectional (Bidirectional (None, 50, 100)           46000     
    _________________________________________________________________
    global_max_pooling1d (Global (None, 100)               0         
    _________________________________________________________________
    dropout (Dropout)            (None, 100)               0         
    _________________________________________________________________
    dense (Dense)                (None, 16)                1616      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 3)                 51        
    =================================================================
    Total params: 367,667
    Trainable params: 367,667
    Non-trainable params: 0
    _________________________________________________________________
    


```python
from keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor="val_loss",patience=1)
hist=model_1.fit(xtrain_pad ,y=ytrain_enc, batch_size=32, epochs=100, validation_data=(xvalid_pad, yvalid_enc), callbacks=[early_stop])
```

    Epoch 1/100
    429/429 [==============================] - 11s 26ms/step - loss: 0.8356 - accuracy: 0.6025 - val_loss: 0.5530 - val_accuracy: 0.7821
    Epoch 2/100
    429/429 [==============================] - 11s 26ms/step - loss: 0.4419 - accuracy: 0.8333 - val_loss: 0.5059 - val_accuracy: 0.7967
    Epoch 3/100
    429/429 [==============================] - 11s 26ms/step - loss: 0.3187 - accuracy: 0.8845 - val_loss: 0.5161 - val_accuracy: 0.7998
    


```python
vloss=hist.history['val_loss']
loss=hist.history['loss']

x_len=np.arange(len(loss))

plt.plot(x_len, vloss, marker=".", color='r', label='val_loss')
plt.plot(x_len,loss,marker=".",color="b",label='loss')
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
```

![output_46_0](https://user-images.githubusercontent.com/45659433/119966797-a06fd900-bfe6-11eb-8e30-f298d99e63d3.png)


## 모델2 : LSTM 조정


```python
from keras.layers.recurrent import LSTM, GRU
from keras.layers import GlobalAveragePooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
```


```python
# A simple LSTM with glove embeddings and two dense layers
model_2 = Sequential()
model_2.add(Embedding(num_words, emb_size,input_length=max_len))
model_2.add(SpatialDropout1D(0.3))
model_2.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model_2.add(Dense(1024, activation='relu'))
model_2.add(Dropout(0.8))

model_2.add(Dense(1024, activation='relu'))
model_2.add(Dropout(0.8))

model_2.add(Dense(3))
model_2.add(Activation('softmax'))
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
hist1=model_2.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])
```

    Epoch 1/100
    27/27 [==============================] - 6s 231ms/step - loss: 1.0755 - accuracy: 0.4058 - val_loss: 1.0459 - val_accuracy: 0.4780
    Epoch 2/100
    27/27 [==============================] - 6s 214ms/step - loss: 0.9246 - accuracy: 0.5598 - val_loss: 0.7514 - val_accuracy: 0.6868
    Epoch 3/100
    27/27 [==============================] - 6s 207ms/step - loss: 0.6162 - accuracy: 0.7534 - val_loss: 0.5447 - val_accuracy: 0.7840
    Epoch 4/100
    27/27 [==============================] - 6s 212ms/step - loss: 0.4445 - accuracy: 0.8306 - val_loss: 0.4921 - val_accuracy: 0.8047
    Epoch 5/100
    27/27 [==============================] - 6s 227ms/step - loss: 0.3621 - accuracy: 0.8627 - val_loss: 0.4945 - val_accuracy: 0.8032
    Epoch 6/100
    27/27 [==============================] - 6s 215ms/step - loss: 0.3134 - accuracy: 0.8808 - val_loss: 0.4965 - val_accuracy: 0.8109
    Epoch 7/100
    27/27 [==============================] - 6s 215ms/step - loss: 0.2718 - accuracy: 0.8989 - val_loss: 0.5272 - val_accuracy: 0.8069
    


```python
model_2.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 50, 64)            320000    
    _________________________________________________________________
    spatial_dropout1d (SpatialDr (None, 50, 64)            0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               66000     
    _________________________________________________________________
    dense_2 (Dense)              (None, 1024)              103424    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1024)              1049600   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 3)                 3075      
    _________________________________________________________________
    activation (Activation)      (None, 3)                 0         
    =================================================================
    Total params: 1,542,099
    Trainable params: 1,542,099
    Non-trainable params: 0
    _________________________________________________________________
    


```python
vloss=hist1.history['val_loss']
loss=hist1.history['loss']

x_len=np.arange(len(loss))

plt.plot(x_len, vloss, marker=".", color='r', label='val_loss')
plt.plot(x_len,loss,marker=".",color="b",label='loss')
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
```


![output_52_0](https://user-images.githubusercontent.com/45659433/119966802-a1086f80-bfe6-11eb-82ea-57a82024b053.png)


## 모델3: GRU


```python
# GRU with glove embeddings and two dense layers

model_3 = Sequential()
model_3.add(Embedding(num_words, emb_size,input_length=max_len))
model_3.add(SpatialDropout1D(0.3))
model_3.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model_3.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model_3.add(Dense(1024, activation='relu'))
model_3.add(Dropout(0.8))

model_3.add(Dense(1024, activation='relu'))
model_3.add(Dropout(0.8))

model_3.add(Dense(3))
model_3.add(Activation('softmax'))
model_3.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
hist2=model_3.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])
```

    Epoch 1/100
    27/27 [==============================] - 42s 2s/step - loss: 1.0754 - accuracy: 0.4089 - val_loss: 1.0350 - val_accuracy: 0.4721
    Epoch 2/100
    27/27 [==============================] - 42s 2s/step - loss: 0.9275 - accuracy: 0.5489 - val_loss: 0.8199 - val_accuracy: 0.6248
    Epoch 3/100
    27/27 [==============================] - 45s 2s/step - loss: 0.7017 - accuracy: 0.7100 - val_loss: 0.6140 - val_accuracy: 0.7487
    Epoch 4/100
    27/27 [==============================] - 50s 2s/step - loss: 0.5195 - accuracy: 0.7989 - val_loss: 0.5454 - val_accuracy: 0.7831
    Epoch 5/100
    27/27 [==============================] - 49s 2s/step - loss: 0.4270 - accuracy: 0.8411 - val_loss: 0.5122 - val_accuracy: 0.7933
    Epoch 6/100
    27/27 [==============================] - 50s 2s/step - loss: 0.3511 - accuracy: 0.8704 - val_loss: 0.5214 - val_accuracy: 0.7952
    Epoch 7/100
    27/27 [==============================] - 52s 2s/step - loss: 0.3088 - accuracy: 0.8850 - val_loss: 0.4973 - val_accuracy: 0.8051
    Epoch 8/100
    27/27 [==============================] - 51s 2s/step - loss: 0.2672 - accuracy: 0.9008 - val_loss: 0.5334 - val_accuracy: 0.8068
    Epoch 9/100
    27/27 [==============================] - 49s 2s/step - loss: 0.2485 - accuracy: 0.9084 - val_loss: 0.5456 - val_accuracy: 0.7964
    Epoch 10/100
    27/27 [==============================] - 51s 2s/step - loss: 0.2452 - accuracy: 0.9119 - val_loss: 0.5676 - val_accuracy: 0.7932
    


```python
model_3.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 50, 64)            320000    
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 50, 64)            0         
    _________________________________________________________________
    gru (GRU)                    (None, 50, 300)           329400    
    _________________________________________________________________
    gru_1 (GRU)                  (None, 300)               541800    
    _________________________________________________________________
    dense_5 (Dense)              (None, 1024)              308224    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1024)              1049600   
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 3)                 3075      
    _________________________________________________________________
    activation_1 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 2,552,099
    Trainable params: 2,552,099
    Non-trainable params: 0
    _________________________________________________________________
    


```python
vloss=hist2.history['val_loss']
loss=hist2.history['loss']

x_len=np.arange(len(loss))

plt.plot(x_len, vloss, marker=".", color='r', label='val_loss')
plt.plot(x_len,loss,marker=".",color="b",label='loss')
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
```


![output_56_0](https://user-images.githubusercontent.com/45659433/119966803-a1086f80-bfe6-11eb-8d6c-9d44e1d50b2d.png)


## 6. 모델 선택


```python
model_1.evaluate(xvalid_pad,yvalid_enc)
```

    184/184 [==============================] - 1s 4ms/step - loss: 0.5161 - accuracy: 0.7998
    




    [0.5161198973655701, 0.7997956871986389]




```python
model_2.evaluate(xvalid_pad,yvalid_enc)
```

    184/184 [==============================] - 1s 7ms/step - loss: 0.5272 - accuracy: 0.8069
    




    [0.5272414684295654, 0.8069458603858948]




```python
model_3.evaluate(xvalid_pad,yvalid_enc)
```

    184/184 [==============================] - 9s 48ms/step - loss: 0.5676 - accuracy: 0.7932
    




    [0.5676275491714478, 0.7931562662124634]



적당한 파라미터 수와 loss값 고려 -> model_1를 선택


```python
results=model_1.predict(test_pad)
ids= test['id']
result=pd.DataFrame(results, columns=["EAP","HPL","MWS"])
result.insert(0,"id",ids)
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>EAP</th>
      <th>HPL</th>
      <th>MWS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id02310</td>
      <td>0.159981</td>
      <td>0.016428</td>
      <td>0.823591</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id24541</td>
      <td>0.991916</td>
      <td>0.003873</td>
      <td>0.004211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id00134</td>
      <td>0.054596</td>
      <td>0.940245</td>
      <td>0.005159</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id27757</td>
      <td>0.979490</td>
      <td>0.009514</td>
      <td>0.010995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id04081</td>
      <td>0.853191</td>
      <td>0.095087</td>
      <td>0.051722</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('submit.csv',index=False)
```

## 추가. 자연어처리 전처리 분석관련

- CountVectorizer:

문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 인코딩 벡터를 만든다.

- TfidfVectorizer:

CountVectorizer와 비슷하지만 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 인코딩 벡터를 만든다.


```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
```

### CountVectorizer

1) 문서를 토큰 리스트로 변환한다.

2) 각 문서에서 토큰의 출현 빈도를 센다.

3) 각 문서를 BOW 인코딩 벡터로 변환한다.

- 기본 


```python
vect=CountVectorizer()
vect.fit((list(x_train) + list(x_valid)))
vect.vocabulary_
```




    {'you': 24991,
     'came': 3046,
     'the': 22085,
     'embodied': 7290,
     'image': 11030,
     'of': 15145,
     'my': 14491,
     'fondest': 8910,
     'dreams': 6755,
     'creator': 5012,
     'abhor': 32,
     'me': 13678,
     'what': 24493,
     'hope': 10702,
     'can': 3066,
     'gather': 9450,
     'from': 9207,
     'your': 24998,
     'fellow': 8494,
     'creatures': 5014,
     'who': 24593,
     'owe': 15560,
     'nothing': 14906,
     'well': 24459,
     'remember': 18258,
     'it': 12093,
     'had': 10097,
     'no': 14817,
     'trees': 22694,
     'nor': 14868,
     'benches': 2103,
     'anything': 982,
     'similar': 20067,
     'within': 24741,
     'especially': 7753,
     'there': 22118,
     'is': 12063,
     'to': 22393,
     'be': 1926,
     'made': 13318,
     'in': 11236,
     'this': 22175,
     'way': 24375,
     'without': 24742,
     'method': 13891,
     'true': 22820,
     'wretchedness': 24881,
     'indeed': 11370,
     'ultimate': 22981,
     'woe': 24758,
     'particular': 15800,
     'not': 14898,
     'diffuse': 6100,
     'barzai': 1868,
     'and': 837,
     'atal': 1438,
     'went': 24465,
     'out': 15407,
     'hatheg': 10274,
     'into': 11898,
     'stony': 21035,
     'desert': 5863,
     'despite': 5914,
     'prayers': 16871,
     'peasants': 15952,
     'talked': 21782,
     'earth': 6987,
     'gods': 9728,
     'by': 2955,
     'their': 22093,
     'campfires': 3060,
     'at': 1437,
     'night': 14775,
     'infancy': 11497,
     'was': 24326,
     'noted': 14903,
     'for': 8941,
     'docility': 6555,
     'humanity': 10815,
     'disposition': 6406,
     'then': 22099,
     'bank': 1782,
     'defaulter': 5578,
     'remembered': 18260,
     'picture': 16342,
     'suggested': 21366,
     'that': 22080,
     'viewed': 24011,
     'filed': 8606,
     'identification': 10952,
     'police': 16652,
     'headquarters': 10328,
     'saw': 19252,
     'moat': 14107,
     'filled': 8614,
     'some': 20460,
     'known': 12473,
     'towers': 22527,
     'were': 24467,
     'demolished': 5717,
     'whilst': 24535,
     'new': 14731,
     'wings': 24689,
     'existed': 8030,
     'confuse': 4417,
     'beholder': 2053,
     'time': 22334,
     'his': 10585,
     'pulse': 17441,
     'imperceptible': 11124,
     'breathing': 2663,
     'stertorous': 20977,
     'intervals': 11879,
     'half': 10120,
     'minute': 14010,
     'above': 65,
     'nighted': 14777,
     'screaming': 19405,
     'men': 13793,
     'horses': 10741,
     'dæmonic': 6961,
     'drumming': 6840,
     'rose': 18886,
     'louder': 13168,
     'pitch': 16435,
     'an': 797,
     'ice': 10933,
     'cold': 4009,
     'wind': 24665,
     'shocking': 19874,
     'sentience': 19630,
     'deliberateness': 5648,
     'swept': 21649,
     'down': 6679,
     'those': 22189,
     'forbidden': 8948,
     'heights': 10417,
     'coiled': 3999,
     'about': 64,
     'each': 6963,
     'man': 13440,
     'separately': 19642,
     'till': 22322,
     'all': 632,
     'cohort': 3997,
     'struggling': 21188,
     'dark': 5366,
     'as': 1287,
     'if': 10981,
     'acting': 246,
     'fate': 8386,
     'laocoön': 12593,
     'sons': 20488,
     'arms': 1213,
     'stirred': 21018,
     'disquietingly': 6419,
     'legs': 12788,
     'drew': 6770,
     'up': 23641,
     'various': 23827,
     'muscles': 14435,
     'contracted': 4648,
     'repulsive': 18410,
     'kind': 12395,
     'writhing': 24900,
     'arthur': 1261,
     'munroe': 14415,
     'dead': 5422,
     'alas': 578,
     'how': 10783,
     'great': 9882,
     'contrast': 4659,
     'between': 2200,
     'us': 23693,
     'he': 10316,
     'alive': 630,
     'every': 7852,
     'scene': 19305,
     'joyful': 12251,
     'when': 24508,
     'beauties': 1966,
     'setting': 19704,
     'sun': 21414,
     'more': 14248,
     'happy': 10192,
     'beheld': 2049,
     'rise': 18776,
     'recommence': 17985,
     'day': 5408,
     'adventure': 374,
     'occurred': 15107,
     'near': 14620,
     'richmond': 18703,
     'virginia': 24067,
     'grey': 9912,
     'headed': 10320,
     'ye': 24952,
     'hoped': 10703,
     'yet': 24975,
     'few': 8554,
     'years': 24960,
     'long': 13112,
     'abode': 51,
     'but': 2931,
     'lease': 12733,
     'must': 14459,
     'remove': 18291,
     'children': 3608,
     'will': 24649,
     'never': 14726,
     'reach': 17848,
     'maturity': 13658,
     'even': 7833,
     'now': 14944,
     'small': 20295,
     'grave': 9857,
     'dug': 6874,
     'mothers': 14304,
     'clasp': 3806,
     'them': 22095,
     'one': 15229,
     'death': 5445,
     'embraces': 7300,
     'shuddering': 19965,
     'stretched': 21130,
     'hands': 10159,
     'eyes': 8203,
     'cast': 3254,
     'seemed': 19553,
     'bursting': 2905,
     'sockets': 20397,
     'while': 24532,
     'appeared': 1040,
     'follow': 8901,
     'shapes': 19782,
     'invisible': 11981,
     'yielding': 24981,
     'air': 558,
     'they': 22133,
     'are': 1166,
     'cried': 5044,
     'shrouds': 19953,
     'pass': 15825,
     'silent': 20047,
     'procession': 17141,
     'towards': 22522,
     'far': 8338,
     'land': 12563,
     'doom': 6629,
     'bloodless': 2390,
     'lips': 12972,
     'move': 14356,
     'shadowy': 19744,
     'limbs': 12930,
     'void': 24139,
     'motion': 14305,
     'still': 21000,
     'glide': 9653,
     'onwards': 15242,
     'did': 6068,
     'entirely': 7598,
     'know': 12466,
     'fatal': 8382,
     'effects': 7098,
     'miserable': 14050,
     'deformity': 5614,
     'dare': 5361,
     'look': 13125,
     'piece': 16349,
     'strange': 21092,
     'jewellery': 12185,
     'said': 19088,
     'miskatonic': 14060,
     'university': 23354,
     'museum': 14441,
     'instant': 11731,
     'afterwards': 479,
     'stood': 21036,
     'with': 24726,
     'living': 13029,
     'child': 3601,
     'grasp': 9839,
     'upon': 23655,
     'marble': 13514,
     'flagstones': 8720,
     'side': 19998,
     'marchesa': 13520,
     'cloak': 3888,
     'heavy': 10386,
     'drenching': 6761,
     'water': 24349,
     'became': 1975,
     'unfastened': 23245,
     'falling': 8282,
     'folds': 8893,
     'feet': 8480,
     'discovered': 6275,
     'wonder': 24776,
     'stricken': 21138,
     'spectators': 20632,
     'graceful': 9795,
     'person': 16166,
     'very': 23955,
     'young': 24992,
     'sound': 20535,
     'whose': 24603,
     'name': 14533,
     'greater': 9883,
     'part': 15781,
     'europe': 7821,
     'ringing': 18756,
     'have': 10293,
     'already': 703,
     'put': 17531,
     'off': 15146,
     'carelessness': 3183,
     'childhood': 3603,
     'unlined': 23378,
     'brow': 2773,
     'springy': 20755,
     'gait': 9354,
     'early': 6977,
     'may': 13669,
     'adorn': 352,
     'thee': 22091,
     'say': 19257,
     'its': 12104,
     'pleasures': 16561,
     'soon': 20490,
     'endeavoured': 7432,
     'extract': 8162,
     'patience': 15876,
     'her': 10469,
     'ideas': 10947,
     'she': 19802,
     'many': 13505,
     'crowded': 5094,
     'family': 8307,
     'degenerated': 5621,
     'moved': 14358,
     'across': 242,
     'valley': 23778,
     'merged': 13840,
     'mongrel': 14173,
     'population': 16723,
     'which': 24527,
     'later': 12636,
     'produce': 17159,
     'pitiful': 16448,
     'squatters': 20792,
     'convinced': 4726,
     'wilcox': 24634,
     'older': 15191,
     'matters': 13651,
     'mentioned': 13817,
     'professor': 17180,
     'bought': 2565,
     'automatic': 1587,
     'almost': 684,
     'took': 22447,
     'step': 20961,
     'certain': 3403,
     'deterred': 5970,
     'seeing': 19544,
     'however': 10784,
     'precise': 16901,
     'counterpart': 4886,
     'felon': 8499,
     'standing': 20860,
     'upright': 23661,
     'cart': 3221,
     'before': 2019,
     'so': 20373,
     'expressed': 8121,
     'themselves': 22098,
     'having': 10298,
     'communicated': 4169,
     'opinion': 15275,
     'another': 919,
     'dram': 6708,
     'knocked': 12456,
     'butt': 2937,
     'ends': 7446,
     'muskets': 14457,
     'henry': 10466,
     'wheeler': 24502,
     'trembling': 22704,
     'turned': 22897,
     'rescued': 18429,
     'telescope': 21915,
     'on': 15225,
     'mountain': 14332,
     'see': 19535,
     'might': 13930,
     'completion': 4237,
     'demoniacal': 5720,
     'design': 5874,
     'insatiable': 11667,
     'passion': 15837,
     'honour': 10685,
     'since': 20087,
     'tyrant': 22967,
     'syracuse': 21728,
     'would': 24845,
     'work': 24806,
     'none': 14855,
     'save': 19242,
     'him': 10560,
     'or': 15303,
     'kalos': 12328,
     'lightning': 12910,
     'flashed': 8736,
     'again': 482,
     'somewhat': 20470,
     'brighter': 2710,
     'than': 22069,
     'crowd': 5093,
     'fancied': 8316,
     'shewed': 19835,
     'mistiness': 14081,
     'around': 1220,
     'altar': 707,
     'stone': 21032,
     'distant': 6462,
     'height': 10413,
     'we': 24384,
     'rapidly': 17778,
     'length': 12797,
     'number': 14963,
     'forms': 9023,
     'could': 4865,
     'discerned': 6226,
     'sides': 20002,
     'grew': 9911,
     'big': 2236,
     'splash': 20704,
     'oars': 14999,
     'audible': 1534,
     'distinguish': 6480,
     'languid': 12585,
     'form': 9010,
     'friend': 9173,
     'raised': 17729,
     'himself': 10561,
     'our': 15402,
     'approach': 1091,
     'dilemma': 6124,
     'captain': 3145,
     'hardy': 10213,
     'arranged': 1228,
     'corpse': 4806,
     'being': 2058,
     'first': 8672,
     'partially': 15793,
     'embalmed': 7268,
     'packed': 15595,
     'large': 12608,
     'quantity': 17575,
     'salt': 19118,
     'box': 2594,
     'suitable': 21375,
     'dimensions': 6133,
     'should': 19906,
     'conveyed': 4719,
     'board': 2433,
     'merchandise': 13826,
     'everything': 7856,
     'tainted': 21765,
     'loathsome': 13047,
     'contagion': 4586,
     'inspired': 11719,
     'noxious': 14947,
     'alliance': 656,
     'distorted': 6485,
     'hidden': 10532,
     'powers': 16841,
     'arrival': 1241,
     'dear': 5440,
     'cousin': 4925,
     'fills': 8618,
     'drilled': 6776,
     'myself': 14500,
     'preparation': 16965,
     'crucial': 5103,
     'moment': 14158,
     'blind': 2369,
     'training': 22570,
     'saved': 19243,
     'continued': 4635,
     'create': 5004,
     'female': 8502,
     'whom': 24599,
     'live': 13015,
     'interchange': 11819,
     'sympathies': 21710,
     'necessary': 14643,
     'daemon': 5298,
     'flash': 8735,
     'monstrous': 14200,
     'fireball': 8652,
     'sleeper': 20223,
     'started': 20880,
     'suddenly': 21341,
     'glare': 9631,
     'beyond': 2214,
     'window': 24675,
     'threw': 22227,
     'shadow': 19740,
     'vividly': 24124,
     'chimney': 3626,
     'fireplace': 8656,
     'strayed': 21109,
     'spoke': 20725,
     'outstretched': 15464,
     'hand': 10146,
     'winning': 24695,
     'voice': 24135,
     'turning': 22898,
     'invaders': 11947,
     'severe': 19724,
     'commanded': 4116,
     'lay': 12690,
     'do': 6551,
     'think': 22153,
     'because': 1976,
     'wasted': 24334,
     'plague': 16475,
     'overcome': 15483,
     'also': 705,
     'among': 770,
     'vanquished': 23806,
     'famine': 8308,
     'disease': 6298,
     'ghosts': 9559,
     'murdered': 14419,
     'arise': 1188,
     'bid': 2228,
     'haff': 10104,
     'em': 7260,
     'clean': 3831,
     'gone': 9747,
     'nigh': 14774,
     'left': 12765,
     'sucked': 21337,
     'most': 14298,
     'dry': 6850,
     'blood': 2388,
     'sores': 20512,
     'like': 12914,
     'ben': 2100,
     'whateley': 24494,
     'cattle': 3300,
     'ever': 7843,
     'senct': 19605,
     'lavinny': 12675,
     'black': 2295,
     'brat': 2628,
     'born': 2532,
     'ascendancy': 1292,
     'interposed': 11853,
     'depend': 5773,
     'robber': 18822,
     'knowledge': 12472,
     'loser': 13154,
     'folks': 8898,
     'three': 22222,
     'darters': 5380,
     'wearin': 24405,
     'gold': 9740,
     'things': 22150,
     'nobody': 14827,
     'afore': 465,
     'smoke': 20316,
     'comin': 4112,
     'aout': 990,
     'refin': 18076,
     'ry': 19037,
     'chimbly': 3618,
     'violent': 24060,
     'sudden': 21340,
     'shock': 19872,
     'through': 22251,
     'soul': 20531,
     'electricity': 7185,
     'poor': 16705,
     'rich': 18695,
     'equal': 7676,
     'rather': 17808,
     'superior': 21465,
     'entered': 7571,
     'such': 21336,
     'tasks': 21850,
     'alacrity': 571,
     'experience': 8068,
     'ignorance': 10991,
     'inaptitude': 11257,
     'habits': 10089,
     'repose': 18373,
     'rendered': 18301,
     'fatiguing': 8403,
     'luxurious': 13274,
     'galling': 9372,
     'proud': 17341,
     'disgustful': 6321,
     'minds': 13983,
     'bent': 2131,
     'intellectual': 11787,
     'improvement': 11219,
     'held': 10426,
     'dearest': 5442,
     'privilege': 17115,
     'exempt': 7986,
     'attending': 1494,
     'mere': 13836,
     'animal': 881,
     'wants': 24292,
     'seldom': 19574,
     'palace': 15636,
     'assured': 1404,
     'public': 17409,
     'duties': 6930,
     'prevent': 17044,
     'remaining': 18245,
     'alone': 688,
     'perdita': 16061,
     'dream': 6746,
     'cognizant': 3991,
     'escape': 7737,
     'only': 15237,
     'single': 20102,
     'impediment': 11111,
     'wheel': 24500,
     'route': 18924,
     'chip': 3639,
     'twig': 22925,
     'feared': 8434,
     'outsiders': 15459,
     'slowly': 20276,
     'accustomed': 194,
     'finally': 8626,
     'helping': 10444,
     'vastly': 23842,
     'beat': 1952,
     'thickets': 22140,
     'tore': 22468,
     'partitions': 15810,
     'mansion': 13484,
     'search': 19469,
     'lurking': 13254,
     'fear': 8433,
     'word': 24799,
     'drawn': 6737,
     'difficulty': 6097,
     'real': 17869,
     'anguish': 875,
     'painted': 15628,
     'features': 8458,
     'lifted': 12893,
     'horror': 10734,
     'fixed': 8701,
     'dread': 6739,
     'ground': 9958,
     'winter': 24697,
     'has': 10250,
     'been': 2009,
     'passed': 15830,
     'miserably': 14051,
     'tortured': 22490,
     'anxious': 974,
     'suspense': 21587,
     'peace': 15935,
     'countenance': 4876,
     'find': 8629,
     'heart': 10353,
     'totally': 22499,
     'comfort': 4103,
     'tranquillity': 22587,
     'feeling': 8475,
     'gave': 9461,
     'place': 16467,
     'irritation': 12062,
     'dreaded': 6740,
     'scream': 19402,
     'uncle': 23087,
     'knew': 12444,
     'against': 483,
     'menace': 13794,
     'defend': 5587,
     'short': 19898,
     'previous': 17049,
     'commencement': 4126,
     'game': 9388,
     'automaton': 1591,
     'wound': 24848,
     'exhibiter': 8009,
     'usual': 23712,
     'ear': 6969,
     'any': 977,
     'degree': 5629,
     'sounds': 20541,
     'produced': 17160,
     'winding': 24670,
     'system': 21733,
     'machinery': 13307,
     'fail': 8246,
     'discover': 6273,
     'instantaneously': 11733,
     'axis': 1670,
     'key': 12369,
     'chess': 3575,
     'player': 16529,
     'cannot': 3095,
     'possibly': 16790,
     'connected': 4458,
     'either': 7143,
     'weight': 24445,
     'spring': 20751,
     'whatever': 24496,
     'told': 22415,
     'elixir': 7220,
     'eternal': 7791,
     'life': 12886,
     'after': 474,
     'began': 2025,
     'call': 3023,
     'slow': 20275,
     'sailing': 19091,
     'stars': 20878,
     'fancy': 8319,
     'glided': 9654,
     'regretfully': 18148,
     'sight': 20023,
     'vision': 24091,
     'opened': 15256,
     'secret': 19507,
     'vistas': 24110,
     'existence': 8031,
     'common': 4156,
     'eye': 8194,
     'suspects': 21582,
     'servant': 19681,
     'course': 4908,
     'fool': 8916,
     'am': 728,
     'sooner': 20491,
     'thought': 22192,
     'obvious': 15085,
     'solution': 20454,
     'repaired': 18324,
     'list': 12982,
     'here': 10478,
     'distinctly': 6478,
     'come': 4091,
     'party': 15819,
     'although': 722,
     'fact': 8229,
     'original': 15346,
     'bring': 2725,
     'words': 24802,
     'written': 24904,
     'overscored': 15520,
     'copied': 4756,
     'formula': 9024,
     'chose': 3681,
     'dr': 6695,
     'armitage': 1208,
     'looked': 13126,
     'involuntarily': 11991,
     'over': 15474,
     'shoulder': 19907,
     'open': 15255,
     'pages': 15612,
     'latin': 12644,
     'version': 23946,
     'contained': 4589,
     'threats': 22221,
     'sanity': 19159,
     'world': 24816,
     'often': 15174,
     'ask': 1317,
     'pure': 17480,
     'phantasm': 16252,
     'freak': 9126,
     'fever': 8549,
     'raving': 17831,
     'boat': 2441,
     'german': 9536,
     'war': 24293,
     'raymond': 17842,
     'strode': 21167,
     'angrily': 873,
     'room': 18873,
     'oh': 15178,
     'thet': 22132,
     'afriky': 472,
     'book': 2501,
     'daily': 5307,
     'vows': 24186,
     'revenge': 18608,
     'deep': 5565,
     'deadly': 5427,
     'compensate': 4211,
     'outrages': 15444,
     'endured': 7452,
     'weariness': 24406,
     'solitude': 20447,
     'depressed': 5808,
     'spirits': 20698,
     'infinitude': 11527,
     'shrieking': 19934,
     'twilight': 22927,
     'abysses': 114,
     'past': 15847,
     'second': 19501,
     'muddy': 14379,
     'unknown': 23365,
     'alley': 654,
     'foetid': 8880,
     'odours': 15140,
     'rotting': 18901,
     'walls': 24262,
     'ancient': 834,
     'houses': 10773,
     'towering': 22526,
     'let': 12824,
     'composition': 4257,
     'defective': 5582,
     'emendation': 7307,
     'wrought': 24908,
     'arrangement': 1229,
     'submitted': 21276,
     'artist': 1276,
     'necessity': 14647,
     'admitted': 332,
     'succession': 21327,
     'loud': 13167,
     'shrill': 19937,
     'screams': 19407,
     'throat': 22236,
     'chained': 3424,
     'thrust': 22260,
     'violently': 24061,
     'back': 1706,
     'calculation': 3013,
     'deceived': 5488,
     'friends': 9176,
     'holding': 10635,
     'little': 13011,
     'musical': 14446,
     'levee': 12837,
     'talk': 21780,
     'dupin': 6906,
     'reply': 18366,
     'heard': 10344,
     'fears': 8441,
     'driven': 6787,
     'people': 16037,
     'kilderry': 12385,
     'laughed': 12658,
     'loudly': 13170,
     'these': 22129,
     'vaguest': 23759,
     'wildest': 24640,
     'absurd': 96,
     'character': 3480,
     'whether': 24526,
     'professes': 17174,
     'play': 16527,
     'informed': 11554,
     'sometimes': 20468,
     'throes': 22243,
     'nightmare': 14781,
     'unseen': 23532,
     'whirl': 24556,
     'roofs': 18871,
     'cities': 3756,
     'toward': 22521,
     'grinning': 9931,
     'chasm': 3524,
     'nis': 14809,
     'relief': 18219,
     'delight': 5657,
     'shriek': 19932,
     'wildly': 24641,
     'throw': 22253,
     'oneself': 15233,
     'voluntarily': 24156,
     'along': 689,
     'hideous': 10534,
     'vortex': 24173,
     'bottomless': 2558,
     'gulf': 10040,
     'yawn': 24947,
     'exceedingly': 7914,
     'thin': 22147,
     'associates': 1389,
     'asserted': 1363,
     'answered': 921,
     'drunk': 6843,
     'pennant': 16018,
     'mast': 13607,
     'head': 10317,
     'sober': 20385,
     'served': 19684,
     'jib': 12190,
     'boom': 2507,
     'flight': 8784,
     'railroad': 17717,
     'kanadaw': 12329,
     'continent': 4626,
     'lengthened': 12799,
     'tour': 22511,
     'scotland': 19377,
     'loch': 13061,
     'katrine': 12338,
     'lomond': 13103,
     'thence': 22100,
     'crossed': 5082,
     'ireland': 12010,
     'several': 19722,
     'weeks': 24437,
     'neighbourhood': 14679,
     'killarney': 12387,
     'individual': 11429,
     'thinking': 22157,
     'beings': 2059,
     'incarnate': 11271,
     'portions': 16755,
     'divine': 6534,
     'mind': 13979,
     'town': 22528,
     'able': 44,
     'inspire': 11718,
     'dislike': 6346,
     'neighbours': 14682,
     'least': 12734,
     'unusual': 23600,
     'worthy': 24844,
     'tourist': 22512,
     'attention': 1496,
     'accident': 133,
     'interminable': 11842,
     'writhings': 24901,
     'jarred': 12138,
     'forgotten': 9005,
     'electric': 7182,
     'lamp': 12558,
     'alight': 622,
     'shone': 19883,
     'eerily': 7086,
     'burrow': 2901,
     'caked': 3002,
     'loam': 13042,
     'curved': 5235,
     'ahead': 533,
     'fine': 8633,
     'despair': 5896,
     'committed': 4149,
     'matter': 13649,
     'amid': 765,
     'perfect': 16068,
     'whirlwind': 24562,
     'sagacious': 19082,
     'agent': 490,
     'suppose': 21503,
     'desired': 5883,
     'imagined': 11040,
     'state': 20891,
     'rooms': 18876,
     'sufficiently': 21356,
     'roomy': 18877,
     'two': 22947,
     'berths': 2151,
     'other': 15391,
     'landed': 12565,
     'proceeded': 17135,
     'paris': 15759,
     'rumour': 19002,
     'adrian': 358,
     'become': 1981,
     'write': 24894,
     'mad': 13311,
     'lord': 13148,
     'favourite': 8426,
     'ex': 7881,
     'queen': 17594,
     'daughter': 5398,
     'destined': 5930,
     'husband': 10875,
     'risen': 18777,
     'blackness': 2307,
     'twenty': 22922,
     'seven': 19716,
     'centuries': 3389,
     'messages': 13865,
     'places': 16470,
     'planet': 16492,
     'got': 9772,
     'old': 15188,
     'begun': 2042,
     ...}




```python
from collections import Counter
counts=Counter(vect.vocabulary_)
tags=counts.most_common(40)
tags
```




    [('υπνος', 25067),
     ('οἶδα', 25066),
     ('émeutes', 25065),
     ('élite', 25064),
     ('æschylus', 25063),
     ('ærostation', 25062),
     ('æronauts', 25061),
     ('æronaut', 25060),
     ('ærial', 25059),
     ('æneid', 25058),
     ('æmilianus', 25057),
     ('ægyptus', 25056),
     ('ædile', 25055),
     ('ångstrom', 25054),
     ('zuro', 25053),
     ('zubmizzion', 25052),
     ('zorry', 25051),
     ('zopyrus', 25050),
     ('zones', 25049),
     ('zone', 25048),
     ('zokkar', 25047),
     ('zoilus', 25046),
     ('zodiacal', 25045),
     ('zodiac', 25044),
     ('zobnarian', 25043),
     ('zobna', 25042),
     ('zit', 25041),
     ('zircon', 25040),
     ('zimmerman', 25039),
     ('zimmer', 25038),
     ('zigzagging', 25037),
     ('zigzagged', 25036),
     ('zigzag', 25035),
     ('zide', 25034),
     ('zette', 25033),
     ('zest', 25032),
     ('zerubbabel', 25031),
     ('zephyrs', 25030),
     ('zephyr', 25029),
     ('zenobia', 25028)]



- stopwords를 직접 지정


```python
vect = CountVectorizer(stop_words=["and", "is", "the", "this", 'υπνος','οἶδα']).fit((list(x_train) + list(x_valid)))
vect.vocabulary_
```




    {'you': 24987,
     'came': 3045,
     'embodied': 7289,
     'image': 11029,
     'of': 15143,
     'my': 14489,
     'fondest': 8909,
     'dreams': 6754,
     'creator': 5011,
     'abhor': 32,
     'me': 13676,
     'what': 24489,
     'hope': 10701,
     'can': 3065,
     'gather': 9449,
     'from': 9206,
     'your': 24994,
     'fellow': 8493,
     'creatures': 5013,
     'who': 24589,
     'owe': 15558,
     'nothing': 14904,
     'well': 24455,
     'remember': 18256,
     'it': 12091,
     'had': 10096,
     'no': 14815,
     'trees': 22690,
     'nor': 14866,
     'benches': 2102,
     'anything': 981,
     'similar': 20065,
     'within': 24737,
     'especially': 7752,
     'there': 22115,
     'to': 22389,
     'be': 1925,
     'made': 13316,
     'in': 11235,
     'way': 24371,
     'without': 24738,
     'method': 13889,
     'true': 22816,
     'wretchedness': 24877,
     'indeed': 11369,
     'ultimate': 22977,
     'woe': 24754,
     'particular': 15798,
     'not': 14896,
     'diffuse': 6099,
     'barzai': 1867,
     'atal': 1437,
     'went': 24461,
     'out': 15405,
     'hatheg': 10273,
     'into': 11897,
     'stony': 21033,
     'desert': 5862,
     'despite': 5913,
     'prayers': 16869,
     'peasants': 15950,
     'talked': 21780,
     'earth': 6986,
     'gods': 9727,
     'by': 2954,
     'their': 22090,
     'campfires': 3059,
     'at': 1436,
     'night': 14773,
     'infancy': 11496,
     'was': 24322,
     'noted': 14901,
     'for': 8940,
     'docility': 6554,
     'humanity': 10814,
     'disposition': 6405,
     'then': 22096,
     'bank': 1781,
     'defaulter': 5577,
     'remembered': 18258,
     'picture': 16340,
     'suggested': 21364,
     'that': 22078,
     'viewed': 24007,
     'filed': 8605,
     'identification': 10951,
     'police': 16650,
     'headquarters': 10327,
     'saw': 19250,
     'moat': 14105,
     'filled': 8613,
     'some': 20458,
     'known': 12471,
     'towers': 22523,
     'were': 24463,
     'demolished': 5716,
     'whilst': 24531,
     'new': 14729,
     'wings': 24685,
     'existed': 8029,
     'confuse': 4416,
     'beholder': 2052,
     'time': 22330,
     'his': 10584,
     'pulse': 17439,
     'imperceptible': 11123,
     'breathing': 2662,
     'stertorous': 20975,
     'intervals': 11878,
     'half': 10119,
     'minute': 14008,
     'above': 65,
     'nighted': 14775,
     'screaming': 19403,
     'men': 13791,
     'horses': 10740,
     'dæmonic': 6960,
     'drumming': 6839,
     'rose': 18884,
     'louder': 13166,
     'pitch': 16433,
     'an': 797,
     'ice': 10932,
     'cold': 4008,
     'wind': 24661,
     'shocking': 19872,
     'sentience': 19628,
     'deliberateness': 5647,
     'swept': 21647,
     'down': 6678,
     'those': 22185,
     'forbidden': 8947,
     'heights': 10416,
     'coiled': 3998,
     'about': 64,
     'each': 6962,
     'man': 13438,
     'separately': 19640,
     'till': 22318,
     'all': 632,
     'cohort': 3996,
     'struggling': 21186,
     'dark': 5365,
     'as': 1286,
     'if': 10980,
     'acting': 246,
     'fate': 8385,
     'laocoön': 12591,
     'sons': 20486,
     'arms': 1212,
     'stirred': 21016,
     'disquietingly': 6418,
     'legs': 12786,
     'drew': 6769,
     'up': 23637,
     'various': 23823,
     'muscles': 14433,
     'contracted': 4647,
     'repulsive': 18408,
     'kind': 12393,
     'writhing': 24896,
     'arthur': 1260,
     'munroe': 14413,
     'dead': 5421,
     'alas': 578,
     'how': 10782,
     'great': 9881,
     'contrast': 4658,
     'between': 2199,
     'us': 23689,
     'he': 10315,
     'alive': 630,
     'every': 7851,
     'scene': 19303,
     'joyful': 12249,
     'when': 24504,
     'beauties': 1965,
     'setting': 19702,
     'sun': 21412,
     'more': 14246,
     'happy': 10191,
     'beheld': 2048,
     'rise': 18774,
     'recommence': 17983,
     'day': 5407,
     'adventure': 374,
     'occurred': 15105,
     'near': 14618,
     'richmond': 18701,
     'virginia': 24063,
     'grey': 9911,
     'headed': 10319,
     'ye': 24948,
     'hoped': 10702,
     'yet': 24971,
     'few': 8553,
     'years': 24956,
     'long': 13110,
     'abode': 51,
     'but': 2930,
     'lease': 12731,
     'must': 14457,
     'remove': 18289,
     'children': 3607,
     'will': 24645,
     'never': 14724,
     'reach': 17846,
     'maturity': 13656,
     'even': 7832,
     'now': 14942,
     'small': 20293,
     'grave': 9856,
     'dug': 6873,
     'mothers': 14302,
     'clasp': 3805,
     'them': 22092,
     'one': 15227,
     'death': 5444,
     'embraces': 7299,
     'shuddering': 19963,
     'stretched': 21128,
     'hands': 10158,
     'eyes': 8202,
     'cast': 3253,
     'seemed': 19551,
     'bursting': 2904,
     'sockets': 20395,
     'while': 24528,
     'appeared': 1039,
     'follow': 8900,
     'shapes': 19780,
     'invisible': 11980,
     'yielding': 24977,
     'air': 558,
     'they': 22130,
     'are': 1165,
     'cried': 5043,
     'shrouds': 19951,
     'pass': 15823,
     'silent': 20045,
     'procession': 17139,
     'towards': 22518,
     'far': 8337,
     'land': 12561,
     'doom': 6628,
     'bloodless': 2389,
     'lips': 12970,
     'move': 14354,
     'shadowy': 19742,
     'limbs': 12928,
     'void': 24135,
     'motion': 14303,
     'still': 20998,
     'glide': 9652,
     'onwards': 15240,
     'did': 6067,
     'entirely': 7597,
     'know': 12464,
     'fatal': 8381,
     'effects': 7097,
     'miserable': 14048,
     'deformity': 5613,
     'dare': 5360,
     'look': 13123,
     'piece': 16347,
     'strange': 21090,
     'jewellery': 12183,
     'said': 19086,
     'miskatonic': 14058,
     'university': 23350,
     'museum': 14439,
     'instant': 11730,
     'afterwards': 479,
     'stood': 21034,
     'with': 24722,
     'living': 13027,
     'child': 3600,
     'grasp': 9838,
     'upon': 23651,
     'marble': 13512,
     'flagstones': 8719,
     'side': 19996,
     'marchesa': 13518,
     'cloak': 3887,
     'heavy': 10385,
     'drenching': 6760,
     'water': 24345,
     'became': 1974,
     'unfastened': 23241,
     'falling': 8281,
     'folds': 8892,
     'feet': 8479,
     'discovered': 6274,
     'wonder': 24772,
     'stricken': 21136,
     'spectators': 20630,
     'graceful': 9794,
     'person': 16164,
     'very': 23951,
     'young': 24988,
     'sound': 20533,
     'whose': 24599,
     'name': 14531,
     'greater': 9882,
     'part': 15779,
     'europe': 7820,
     'ringing': 18754,
     'have': 10292,
     'already': 703,
     'put': 17529,
     'off': 15144,
     'carelessness': 3182,
     'childhood': 3602,
     'unlined': 23374,
     'brow': 2772,
     'springy': 20753,
     'gait': 9353,
     'early': 6976,
     'may': 13667,
     'adorn': 352,
     'thee': 22088,
     'say': 19255,
     'its': 12102,
     'pleasures': 16559,
     'soon': 20488,
     'endeavoured': 7431,
     'extract': 8161,
     'patience': 15874,
     'her': 10468,
     'ideas': 10946,
     'she': 19800,
     'many': 13503,
     'crowded': 5093,
     'family': 8306,
     'degenerated': 5620,
     'moved': 14356,
     'across': 242,
     'valley': 23774,
     'merged': 13838,
     'mongrel': 14171,
     'population': 16721,
     'which': 24523,
     'later': 12634,
     'produce': 17157,
     'pitiful': 16446,
     'squatters': 20790,
     'convinced': 4725,
     'wilcox': 24630,
     'older': 15189,
     'matters': 13649,
     'mentioned': 13815,
     'professor': 17178,
     'bought': 2564,
     'automatic': 1586,
     'almost': 684,
     'took': 22443,
     'step': 20959,
     'certain': 3402,
     'deterred': 5969,
     'seeing': 19542,
     'however': 10783,
     'precise': 16899,
     'counterpart': 4885,
     'felon': 8498,
     'standing': 20858,
     'upright': 23657,
     'cart': 3220,
     'before': 2018,
     'so': 20371,
     'expressed': 8120,
     'themselves': 22095,
     'having': 10297,
     'communicated': 4168,
     'opinion': 15273,
     'another': 918,
     'dram': 6707,
     'knocked': 12454,
     'butt': 2936,
     'ends': 7445,
     'muskets': 14455,
     'henry': 10465,
     'wheeler': 24498,
     'trembling': 22700,
     'turned': 22893,
     'rescued': 18427,
     'telescope': 21913,
     'on': 15223,
     'mountain': 14330,
     'see': 19533,
     'might': 13928,
     'completion': 4236,
     'demoniacal': 5719,
     'design': 5873,
     'insatiable': 11666,
     'passion': 15835,
     'honour': 10684,
     'since': 20085,
     'tyrant': 22963,
     'syracuse': 21726,
     'would': 24841,
     'work': 24802,
     'none': 14853,
     'save': 19240,
     'him': 10559,
     'or': 15301,
     'kalos': 12326,
     'lightning': 12908,
     'flashed': 8735,
     'again': 482,
     'somewhat': 20468,
     'brighter': 2709,
     'than': 22067,
     'crowd': 5092,
     'fancied': 8315,
     'shewed': 19833,
     'mistiness': 14079,
     'around': 1219,
     'altar': 707,
     'stone': 21030,
     'distant': 6461,
     'height': 10412,
     'we': 24380,
     'rapidly': 17776,
     'length': 12795,
     'number': 14961,
     'forms': 9022,
     'could': 4864,
     'discerned': 6225,
     'sides': 20000,
     'grew': 9910,
     'big': 2235,
     'splash': 20702,
     'oars': 14997,
     'audible': 1533,
     'distinguish': 6479,
     'languid': 12583,
     'form': 9009,
     'friend': 9172,
     'raised': 17727,
     'himself': 10560,
     'our': 15400,
     'approach': 1090,
     'dilemma': 6123,
     'captain': 3144,
     'hardy': 10212,
     'arranged': 1227,
     'corpse': 4805,
     'being': 2057,
     'first': 8671,
     'partially': 15791,
     'embalmed': 7267,
     'packed': 15593,
     'large': 12606,
     'quantity': 17573,
     'salt': 19116,
     'box': 2593,
     'suitable': 21373,
     'dimensions': 6132,
     'should': 19904,
     'conveyed': 4718,
     'board': 2432,
     'merchandise': 13824,
     'everything': 7855,
     'tainted': 21763,
     'loathsome': 13045,
     'contagion': 4585,
     'inspired': 11718,
     'noxious': 14945,
     'alliance': 656,
     'distorted': 6484,
     'hidden': 10531,
     'powers': 16839,
     'arrival': 1240,
     'dear': 5439,
     'cousin': 4924,
     'fills': 8617,
     'drilled': 6775,
     'myself': 14498,
     'preparation': 16963,
     'crucial': 5102,
     'moment': 14156,
     'blind': 2368,
     'training': 22566,
     'saved': 19241,
     'continued': 4634,
     'create': 5003,
     'female': 8501,
     'whom': 24595,
     'live': 13013,
     'interchange': 11818,
     'sympathies': 21708,
     'necessary': 14641,
     'daemon': 5297,
     'flash': 8734,
     'monstrous': 14198,
     'fireball': 8651,
     'sleeper': 20221,
     'started': 20878,
     'suddenly': 21339,
     'glare': 9630,
     'beyond': 2213,
     'window': 24671,
     'threw': 22223,
     'shadow': 19738,
     'vividly': 24120,
     'chimney': 3625,
     'fireplace': 8655,
     'strayed': 21107,
     'spoke': 20723,
     'outstretched': 15462,
     'hand': 10145,
     'winning': 24691,
     'voice': 24131,
     'turning': 22894,
     'invaders': 11946,
     'severe': 19722,
     'commanded': 4115,
     'lay': 12688,
     'do': 6550,
     'think': 22150,
     'because': 1975,
     'wasted': 24330,
     'plague': 16473,
     'overcome': 15481,
     'also': 705,
     'among': 770,
     'vanquished': 23802,
     'famine': 8307,
     'disease': 6297,
     'ghosts': 9558,
     'murdered': 14417,
     'arise': 1187,
     'bid': 2227,
     'haff': 10103,
     'em': 7259,
     'clean': 3830,
     'gone': 9746,
     'nigh': 14772,
     'left': 12763,
     'sucked': 21335,
     'most': 14296,
     'dry': 6849,
     'blood': 2387,
     'sores': 20510,
     'like': 12912,
     'ben': 2099,
     'whateley': 24490,
     'cattle': 3299,
     'ever': 7842,
     'senct': 19603,
     'lavinny': 12673,
     'black': 2294,
     'brat': 2627,
     'born': 2531,
     'ascendancy': 1291,
     'interposed': 11852,
     'depend': 5772,
     'robber': 18820,
     'knowledge': 12470,
     'loser': 13152,
     'folks': 8897,
     'three': 22218,
     'darters': 5379,
     'wearin': 24401,
     'gold': 9739,
     'things': 22147,
     'nobody': 14825,
     'afore': 465,
     'smoke': 20314,
     'comin': 4111,
     'aout': 989,
     'refin': 18074,
     'ry': 19035,
     'chimbly': 3617,
     'violent': 24056,
     'sudden': 21338,
     'shock': 19870,
     'through': 22247,
     'soul': 20529,
     'electricity': 7184,
     'poor': 16703,
     'rich': 18693,
     'equal': 7675,
     'rather': 17806,
     'superior': 21463,
     'entered': 7570,
     'such': 21334,
     'tasks': 21848,
     'alacrity': 571,
     'experience': 8067,
     'ignorance': 10990,
     'inaptitude': 11256,
     'habits': 10088,
     'repose': 18371,
     'rendered': 18299,
     'fatiguing': 8402,
     'luxurious': 13272,
     'galling': 9371,
     'proud': 17339,
     'disgustful': 6320,
     'minds': 13981,
     'bent': 2130,
     'intellectual': 11786,
     'improvement': 11218,
     'held': 10425,
     'dearest': 5441,
     'privilege': 17113,
     'exempt': 7985,
     'attending': 1493,
     'mere': 13834,
     'animal': 880,
     'wants': 24288,
     'seldom': 19572,
     'palace': 15634,
     'assured': 1403,
     'public': 17407,
     'duties': 6929,
     'prevent': 17042,
     'remaining': 18243,
     'alone': 688,
     'perdita': 16059,
     'dream': 6745,
     'cognizant': 3990,
     'escape': 7736,
     'only': 15235,
     'single': 20100,
     'impediment': 11110,
     'wheel': 24496,
     'route': 18922,
     'chip': 3638,
     'twig': 22921,
     'feared': 8433,
     'outsiders': 15457,
     'slowly': 20274,
     'accustomed': 194,
     'finally': 8625,
     'helping': 10443,
     'vastly': 23838,
     'beat': 1951,
     'thickets': 22137,
     'tore': 22464,
     'partitions': 15808,
     'mansion': 13482,
     'search': 19467,
     'lurking': 13252,
     'fear': 8432,
     'word': 24795,
     'drawn': 6736,
     'difficulty': 6096,
     'real': 17867,
     'anguish': 874,
     'painted': 15626,
     'features': 8457,
     'lifted': 12891,
     'horror': 10733,
     'fixed': 8700,
     'dread': 6738,
     'ground': 9957,
     'winter': 24693,
     'has': 10249,
     'been': 2008,
     'passed': 15828,
     'miserably': 14049,
     'tortured': 22486,
     'anxious': 973,
     'suspense': 21585,
     'peace': 15933,
     'countenance': 4875,
     'find': 8628,
     'heart': 10352,
     'totally': 22495,
     'comfort': 4102,
     'tranquillity': 22583,
     'feeling': 8474,
     'gave': 9460,
     'place': 16465,
     'irritation': 12061,
     'dreaded': 6739,
     'scream': 19400,
     'uncle': 23083,
     'knew': 12442,
     'against': 483,
     'menace': 13792,
     'defend': 5586,
     'short': 19896,
     'previous': 17047,
     'commencement': 4125,
     'game': 9387,
     'automaton': 1590,
     'wound': 24844,
     'exhibiter': 8008,
     'usual': 23708,
     'ear': 6968,
     'any': 976,
     'degree': 5628,
     'sounds': 20539,
     'produced': 17158,
     'winding': 24666,
     'system': 21731,
     'machinery': 13305,
     'fail': 8245,
     'discover': 6272,
     'instantaneously': 11732,
     'axis': 1669,
     'key': 12367,
     'chess': 3574,
     'player': 16527,
     'cannot': 3094,
     'possibly': 16788,
     'connected': 4457,
     'either': 7142,
     'weight': 24441,
     'spring': 20749,
     'whatever': 24492,
     'told': 22411,
     'elixir': 7219,
     'eternal': 7790,
     'life': 12884,
     'after': 474,
     'began': 2024,
     'call': 3022,
     'slow': 20273,
     'sailing': 19089,
     'stars': 20876,
     'fancy': 8318,
     'glided': 9653,
     'regretfully': 18146,
     'sight': 20021,
     'vision': 24087,
     'opened': 15254,
     'secret': 19505,
     'vistas': 24106,
     'existence': 8030,
     'common': 4155,
     'eye': 8193,
     'suspects': 21580,
     'servant': 19679,
     'course': 4907,
     'fool': 8915,
     'am': 728,
     'sooner': 20489,
     'thought': 22188,
     'obvious': 15083,
     'solution': 20452,
     'repaired': 18322,
     'list': 12980,
     'here': 10477,
     'distinctly': 6477,
     'come': 4090,
     'party': 15817,
     'although': 722,
     'fact': 8228,
     'original': 15344,
     'bring': 2724,
     'words': 24798,
     'written': 24900,
     'overscored': 15518,
     'copied': 4755,
     'formula': 9023,
     ...}



### 불용어 불러오기


```python
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\uos\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
stop_words=stopwords.words('english')
stop_words[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



- stopword를 다운 받은 english에 포함되면 제거 


```python
vect = CountVectorizer(stop_words="english").fit((list(x_train) + list(x_valid)))
vect.vocabulary_
```




    {'came': 2992,
     'embodied': 7215,
     'image': 10905,
     'fondest': 8817,
     'dreams': 6688,
     'creator': 4952,
     'abhor': 32,
     'hope': 10581,
     'gather': 9347,
     'fellow': 8409,
     'creatures': 4954,
     'owe': 15372,
     'remember': 18062,
     'trees': 22437,
     'benches': 2059,
     'similar': 19859,
     'especially': 7676,
     'way': 24106,
     'method': 13747,
     'true': 22563,
     'wretchedness': 24583,
     'ultimate': 22721,
     'woe': 24461,
     'particular': 15610,
     'diffuse': 6036,
     'barzai': 1835,
     ...}



### CountVectorizer()의 이외 옵션 

analyzer="char"

: 알파벳 개수로

token_pattern="t\w+"
: 직접지정한 패턴

tokenizer=nltk.word_tokenize
: 단어 토큰한

## n-gram
- (2,2) : 2개의 연결된 토큰을 한단어로
- (1,2): 1개 또는 2개의 연결된 토큰을 한단어로


```python
vect = CountVectorizer(ngram_range=(2, 2)).fit((list(x_train) + list(x_valid)))
vect.vocabulary_
```




    {'you came': 222734,
     'came the': 31603,
     'the embodied': 182073,
     'embodied image': 54545,
     'image of': 89224,
     'of my': 129305,
     'my fondest': 118572,
     'fondest dreams': 66094,
     'you my': 222939,
     'my creator': 118311,
     'creator abhor': 41641,
     'abhor me': 87,
     'me what': 111278,
     'what hope': 212948,
     'hope can': 86493,
     'can gather': 31737,
     'gather from': 70913,
     'from your': 69797,
     'your fellow': 223399,
     'fellow creatures': 63004,
     'creatures who': 41740,
     'who owe': 215732,
     'owe me': 137174,
     'me nothing': 111038,
     'well remember': 211513,
     'remember it': 151695,
     'it had': 97057,
     'had no': 76413,
     'no trees': 122967,
     'trees nor': 199145,
     'nor benches': 123356,
     'benches nor': 24641,
     'nor anything': 123348,
     'anything similar': 14255,
     'similar within': 163511,
     'within it': 219544,
     'especially there': 56651,
     'there is': 189467,
     'is nothing': 96243,
     'nothing to': 124814,
     'to be': 195208,
     'be made': 21429,
     'made in': 107999,
     'in this': 91822,
     'this way': 192345,
     'way without': 210439,
     'without method': 219710,
     'the true': 187001,
     'true wretchedness': 199706,
     'wretchedness indeed': 221562,
     'indeed the': 92638,
     'the ultimate': 187052,
     'ultimate woe': 200747,
     'woe is': 219897,
     'is particular': 96288,
     'particular not': 138617,
     'not diffuse': 123881,
     'barzai and': 20705,
     'and atal': 7791,
     'atal went': 18698,
     'went out': 211613,
     'out of': 136563,
     'of hatheg': 128520,
     'hatheg into': 78352,
     'into the': 95154,
     'the stony': 186443,
     'stony desert': 171682,
     'desert despite': 46534,
     'despite the': 46942,
     'the prayers': 185134,
     'prayers of': 144357,
     'of peasants': 129560,
     'peasants and': 139667,
     'and talked': 12274,
     'talked of': 176612,
     'of earth': 127961,
     'earth gods': 53036,
     'gods by': 73111,
     'by their': 30933,
     'their campfires': 187827,
     'campfires at': 31656,
     'at night': 18503,
     'from my': 69519,
     'my infancy': 118754,
     'infancy was': 93107,
     'was noted': 209080,
     'noted for': 124696,
     'for the': 66873,
     'the docility': 181845,
     'docility and': 50064,
     'and humanity': 9899,
     'humanity of': 88021,
     'my disposition': 118394,
     'then the': 189244,
     'the bank': 180387,
     'bank defaulter': 20434,
     'defaulter remembered': 45284,
     'remembered the': 151748,
     'the picture': 184941,
     'picture and': 141607,
     'and suggested': 12192,
     'suggested that': 174368,
     'that it': 179046,
     'it be': 96813,
     'be viewed': 21824,
     'viewed and': 206079,
     'and filed': 9348,
     'filed for': 63892,
     'for identification': 66544,
     'identification at': 88583,
     'at police': 18543,
     'police headquarters': 143234,
     'saw that': 157093,
     'that the': 179706,
     'the moat': 184209,
     'moat was': 114207,
     'was filled': 208589,
     'filled in': 63924,
     'in and': 90244,
     'and that': 12332,
     'that some': 179596,
     'some of': 167245,
     'of the': 130574,
     'the well': 187474,
     'well known': 211486,
     'known towers': 100452,
     'towers were': 198408,
     'were demolished': 211867,
     'demolished whilst': 45939,
     'whilst new': 215189,
     'new wings': 121838,
     'wings existed': 217810,
     'existed to': 59093,
     'to confuse': 195406,
     'confuse the': 38666,
     'the beholder': 180472,
     'by this': 30939,
     'this time': 192272,
     'time his': 194613,
     'his pulse': 85462,
     'pulse was': 147071,
     'was imperceptible': 208797,
     'imperceptible and': 89761,
     'and his': 9854,
     'his breathing': 84399,
     'breathing was': 27984,
     'was stertorous': 209543,
     'stertorous and': 171106,
     'and at': 7790,
     'at intervals': 18431,
     'intervals of': 94811,
     'of half': 128499,
     'half minute': 77076,
     'and above': 7486,
     'above the': 429,
     'the nighted': 184449,
     'nighted screaming': 122215,
     'screaming of': 157955,
     'of men': 129135,
     'men and': 111993,
     'and horses': 9880,
     'horses that': 86903,
     'that dæmonic': 178722,
     'dæmonic drumming': 52549,
     'drumming rose': 51962,
     'rose to': 155024,
     'to louder': 196218,
     'louder pitch': 106993,
     'pitch whilst': 141969,
     'whilst an': 215165,
     'an ice': 6815,
     'ice cold': 88435,
     'cold wind': 36762,
     'wind of': 217510,
     'of shocking': 130206,
     'shocking sentience': 161962,
     'sentience and': 159895,
     'and deliberateness': 8672,
     'deliberateness swept': 45620,
     'swept down': 175841,
     'down from': 50873,
     'from those': 69719,
     'those forbidden': 192565,
     'forbidden heights': 67034,
     'heights and': 81370,
     'and coiled': 8311,
     'coiled about': 36670,
     'about each': 247,
     'each man': 52620,
     'man separately': 109004,
     'separately till': 159967,
     'till all': 194422,
     'all the': 4570,
     'the cohort': 181121,
     'cohort was': 36667,
     'was struggling': 209566,
     'struggling and': 172963,
     'and screaming': 11684,
     'screaming in': 157950,
     'in the': 91807,
     'the dark': 181542,
     'dark as': 43342,
     'as if': 16990,
     'if acting': 88736,
     'acting out': 1348,
     'out the': 136594,
     'the fate': 182385,
     'fate of': 61902,
     'of laocoön': 128899,
     'laocoön and': 101140,
     'his sons': 85661,
     'the arms': 180223,
     'arms stirred': 16063,
     'stirred disquietingly': 171522,
     'disquietingly the': 49332,
     'the legs': 183742,
     'legs drew': 102946,
     'drew up': 51678,
     'up and': 202974,
     'and various': 12723,
     'various muscles': 204780,
     'muscles contracted': 117574,
     'contracted in': 39746,
     'in repulsive': 91516,
     'repulsive kind': 152447,
     'kind of': 99809,
     'of writhing': 131041,
     'for arthur': 66275,
     'arthur munroe': 16465,
     'munroe was': 117471,
     'was dead': 208337,
     'alas how': 3864,
     'how great': 87426,
     'great was': 74442,
     'was the': 209648,
     'the contrast': 181292,
     'contrast between': 39794,
     'between us': 25278,
     'us he': 203683,
     'he was': 80266,
     'was alive': 207990,
     'alive to': 4057,
     'to every': 195784,
     'every new': 57812,
     'new scene': 121792,
     'scene joyful': 157569,
     'joyful when': 99020,
     'when he': 213455,
     'he saw': 80089,
     'saw the': 157094,
     'the beauties': 180453,
     'beauties of': 22121,
     'the setting': 185963,
     'setting sun': 160294,
     'sun and': 174588,
     'and more': 10614,
     'more happy': 115396,
     'happy when': 77716,
     'he beheld': 79398,
     'beheld it': 23924,
     'it rise': 97313,
     'rise and': 154169,
     'and recommence': 11396,
     'recommence new': 150395,
     'new day': 121712,
     'this adventure': 191273,
     'adventure occurred': 2154,
     'occurred near': 126559,
     'near richmond': 120672,
     'richmond in': 153844,
     'in virginia': 91948,
     'grey headed': 74927,
     'headed men': 80443,
     'men ye': 112141,
     'ye hoped': 221974,
     'hoped for': 86547,
     'for yet': 66996,
     'yet few': 222391,
     'few years': 63578,
     'years in': 222115,
     'in your': 92047,
     'your long': 223507,
     'long known': 106174,
     'known abode': 100384,
     'abode but': 176,
     'but the': 29984,
     'the lease': 183722,
     'lease is': 102400,
     'is up': 96559,
     'up you': 203162,
     'you must': 222937,
     'must remove': 117854,
     'remove children': 151881,
     'children ye': 34885,
     'ye will': 222018,
     'will never': 217224,
     'never reach': 121566,
     'reach maturity': 149423,
     'maturity even': 110455,
     'even now': 57154,
     'now the': 125486,
     'the small': 186172,
     'small grave': 165269,
     'grave is': 74067,
     ...}



## 빈도
- 토큰의 빈도가 max_df로 지정한 값을 초과 하거나 min_df로 지정한 값보다 작은 경우에는 무시


```python
vect = CountVectorizer(max_df=5000, min_df=10).fit((list(x_train) + list(x_valid)))
vect.vocabulary_, vect.stop_words_
```




    ({'you': 4823,
      'came': 588,
      'image': 2145,
      'my': 2786,
      'dreams': 1258,
      'creator': 941,
      'me': 2610,
      'what': 4680,
      'hope': 2085,
      'can': 590,
      'gather': 1814,
      'from': 1776,
      'your': 4826,
      'fellow': 1624,
      'creatures': 943,
      'who': 4705,
      'nothing': 2870,
      'well': 4671,
      'remember': 3485,
      'it': 2298,
      'had': 1962,
      'no': 2848,
      'trees': 4405,
      'nor': 2860,
      'anything': 194,
      'similar': 3870,
      'within': 4748,
      'especially': 1435,
      'there': 4287,
      'is': 2293,
      'be': 377,
      'made': 2540,
      ...},
      
     {'tissues',
      'magazines',
      'ineradicable',
      'brickwork',
      'homologous',
      'arnheim',
      'reinem',
      'ungrateful',
      'retaliations',
      'etc',
      'wanton',
      'alphonse',
      'portmanteau',
      'condensed',
      'multiplicity',
      'moons',
      'lowered',
      'incredibly',
      'rustics',
      'disfigured',
      'invite',
      'pigs',
      'offices',
      'flailing',
      'lodging',
      'inventive',
      'geometry',
      'douglas',
      'retrospection',
      'dynamically',
      'unrecalled',
      'meditating',
      'financier',
      'whimpered',
      'negligently',
      'irrefutable',
      'mauvais',
      'falona',
      'womenfolks',
      'nun',
      'fireplaces',
      'clogged',
      'refinery',
      'mendicant',
      'furs',
      'heh',
      'suggestive',
      'putrescent',
      'prosier',
      'utica',
      'assaulting',
      'redoubtable',
      'disarranged',
      'revoke',
      'herr',
      'prevail',
      'excites',
      'redness',
      'facilis',
      'fiddle',
      'bout',
      'lastly',
      'feex',
      'expedite',
      'outnumbered',
      'sob',
      'jacques',
      'nascent',
      'controversial',
      'shoals',
      'mallet',
      'oceanographic',
      'festivity',
      'portentous',
      'swarmed',
      'zay',
      'eulogies',
      'besotted',
      'burnished',
      'bitten',
      'canoes',
      'offal',
      'monograph',
      'downfall',
      'imports',
      'mingles',
      'deliriously',
      'donnelly',
      'scythians',
      'complexity',
      'proximate',
      'speeches',
      'sleeps',
      'curtail',
      'lung',
      'objections',
      'shifted',
      'substantial',
      'hangman',
      'scripter',
      'andes',
      'restraint',
      'cxncxrd',
      'curls',
      'heights',
      'anear',
      'moodiness',
      'windless',
      'motley',
      'neurotically',
      'inveterate',
      'veiling',
      'artists',
      'mantled',
      'denunciations',
      'richness',
      'ménageais',
      'answering',
      'stary',
      'hardened',
      'coquetry',
      'bristlin',
      'uncomplaining',
      'indivduationis',
      'tete',
      'clump',
      'transfusion',
      'bleus',
      'cutter',
      'qualm',
      'improvised',
      'perigee',
      'moonlit',
      'madmen',
      'clearness',
      'macassar',
      'nickel',
      'fatalists',
      'loath',
      'sarcophagus',
      'epimenides',
      'brobdignag',
      'lens',
      'musician',
      'magnetics',
      'headstones',
      'splashed',
      'disquietingly',
      'convulsive',
      'apprehensive',
      'simianism',
      'amen',
      'markbrünnen',
      'ermengarde',
      'unlaid',
      'attenuated',
      'lammas',
      'optical',
      'intermarrying',
      'lenity',
      'decypher',
      'farms',
      'stimulates',
      'generosity',
      'vie',
      'lapsed',
      'responses',
      'halo',
      'hips',
      'appreciate',
      'ing',
      'unhand',
      'apprenticeship',
      'sunder',
      'prostrated',
      'loyalty',
      'permeate',
      'whitish',
      'inextricable',
      'socially',
      'cherry',
      'announced',
      'supremeness',
      'teuffel',
      'baleful',
      'ridged',
      'grecians',
      'prefer',
      'sublimity',
      'gushing',
      'inspires',
      'riemannian',
      'loudest',
      'nightmares',
      'statuary',
      'denouement',
      'sacrifices',
      'outsiders',
      'indentures',
      'suspenders',
      'fount',
      'gol',
      'uniform',
      'congregations',
      'aaem',
      'baggy',
      'congeniality',
      'abandoning',
      'insinuating',
      'consternation',
      'fashionable',
      'noiselessly',
      'lend',
      'avowedly',
      'audacities',
      'slew',
      'extremeness',
      'untimed',
      'ami',
      'misguided',
      'loitered',
      'rowels',
      'displease',
      'paine',
      'deodamnatus',
      'deformed',
      'biochemical',
      'plunges',
      'tantalize',
      'coherence',
      'phenomenal',
      'piously',
      'boisterously',
      'soho',
      'handles',
      'catapult',
      'casements',
      'durable',
      'hats',
      'ultras',
      'unacted',
      'mainland',
      'somehaow',
      'nihil',
      'organism',
      'porphyrogene',
      'warehouse',
      'manufacture',
      'xwl',
      'ravishing',
      'brier',
      'linguists',
      'mocked',
      'plungings',
      'incessantly',
      'presburg',
      'avidity',
      'portcullis',
      'puncheon',
      'defer',
      'mangled',
      'shun',
      'petition',
      'uninterruptedly',
      'pecking',
      'valour',
      'rave',
      'rejoices',
      'engendered',
      'windham',
      'ognor',
      'moveable',
      'caboche',
      'growths',
      'seaward',
      'tempestuous',
      'hesitantly',
      ...})




```python
from collections import Counter
counts=Counter(vect.vocabulary_)
tags=counts.most_common(40)
tags
```




    [('zeal', 4833),
     ('zann', 4832),
     ('zadok', 4831),
     ('youthful', 4830),
     ('youth', 4829),
     ('yourself', 4828),
     ('yours', 4827),
     ('your', 4826),
     ('younger', 4825),
     ('young', 4824),
     ('you', 4823),
     ('york', 4822),
     ('yog', 4821),
     ('yielded', 4820),
     ('yield', 4819),
     ('yet', 4818),
     ('yesterday', 4817),
     ('yes', 4816),
     ('yellow', 4815),
     ('years', 4814),
     ('year', 4813),
     ('ye', 4812),
     ('yards', 4811),
     ('yard', 4810),
     ('wyatt', 4809),
     ('wrought', 4808),
     ('wrote', 4807),
     ('wrong', 4806),
     ('written', 4805),
     ('writing', 4804),
     ('writhing', 4803),
     ('writers', 4802),
     ('writer', 4801),
     ('write', 4800),
     ('wrist', 4799),
     ('wrinkled', 4798),
     ('wretchedness', 4797),
     ('wretched', 4796),
     ('wretch', 4795),
     ('wreck', 4794)]



## TF-IDF (Term Frequency -Inverse Documnet Frequency)

- 인코딩은 단어를 갯수 그대로 카운트하지 않고 모든 문서에 공통적으로 들어있는 단어의 경우 문서 구별 능력이 떨어진다고 보아 가중치를 축소


```python
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')
```

- 단어 카운트 가중치를 나타내는 함수로
- min_dfsms DF(문서의 수)의 최소 빈도값 설정.
- analyzer: ;'word' 또는 'char'
- sublinear_tf:TF(단어빈도)가 높을 경우 완만하게 처리하는 효과
- ngram_range : 단어 묶음
- max_features: tf-idf 벡터의 최대 feature를 설정. 단어사전의 인덱스만큼 부여


```python
A_tfidf_sp = tfv.fit_transform(list(x_train) + list(x_valid)) 
tfidf_dict = tfv.get_feature_names()
data_array = A_tfidf_sp.toarray()
data = pd.DataFrame(data_array, columns=tfidf_dict)
data.shape
```




    (19579, 15102)




```python
tfv.fit(list(x_train) + list(x_valid))
tfv.vocabulary_
```




    {'came': 1667,
     'embodied': 4141,
     'image': 6539,
     'dreams': 3833,
     'creator': 2766,
     'abhor': 11,
     'hope': 6313,
     'gather': 5435,
     'fellow': 4901,
     'creatures': 2769,
     'owe': 9382,
     'fellow creatures': 4903,
     'remember': 10847,
     'trees': 13771,
     'benches': 1140,
     'similar': 12078,
     'especially': 4374,
     'way': 14604,
     'method': 8340,
     'true': 13819,
     'wretchedness': 14961,
     'ultimate': 13906,
     'woe': 14856,
     'particular': 9508,
     'barzai': 1006,
     'atal': 771,
     'went': 14655,
     'hatheg': 6043,
     'stony': 12702,
     'desert': 3281,
     'despite': 3312,
     'prayers': 10080,
     'peasants': 9603,
     'talked': 13195,
     ...}



##### 빈도수 높은 단어


```python
from collections import Counter
counts=Counter(tfv.vocabulary_)
tags=counts.most_common(40)
tags
```




    [('zokkar', 15101),
     ('zit', 15100),
     ('zimmer', 15099),
     ('zest', 15098),
     ('zenobia', 15097),
     ('zenith', 15096),
     ('zee', 15095),
     ('zeal', 15094),
     ('zann s', 15093),
     ('zann', 15092),
     ('zaire', 15091),
     ('zadok s', 15090),
     ('zadok allen', 15089),
     ('zadok', 15088),
     ('yxur', 15087),
     ('yxu', 15086),
     ('youths', 15085),
     ('youthful', 15084),
     ('youth seen', 15083),
     ('youth s', 15082),
     ('youth', 15081),
     ('youngest', 15080),
     ('younger days', 15079),
     ('younger', 15078),
     ('young woman', 15077),
     ('young wilcox s', 15076),
     ('young wilcox', 15075),
     ('young sir', 15074),
     ('young people', 15073),
     ('young nobleman', 15072),
     ('young men', 15071),
     ('young man', 15070),
     ('young lady', 15069),
     ('young girl', 15068),
     ('young gentlemen', 15067),
     ('young gentleman s', 15066),
     ('young gentleman', 15065),
     ('young friend', 15064),
     ('young folks', 15063),
     ('young feller', 15062)]



##### 워드클라우드로 시각화


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='white')
cloud = wordcloud.generate_from_frequencies(dict(tags))

plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cloud)
plt.show()
```


![output_93_0](https://user-images.githubusercontent.com/45659433/119966807-a1086f80-bfe6-11eb-871c-756b536bd6bf.png)


#### 참고:

- [kaggle][필사] Spooky Author Identification

- [10주차] 새벽 5시 캐글(kaggle)필사하기-Spooky-author data

- Scikit-Learn의 문서 전처리 기능


```python

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MTIyNjQ1MzddfQ==
-->