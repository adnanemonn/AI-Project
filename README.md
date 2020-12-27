Adnan Hussain 
id:172-115-028

Mrinmoy Das
171-115-243


# Sentiment Analysis using CNN based model Alexnet

Sentiment Analysis is the most common text classification tool that analyses an
incoming comments and determines whether the particular sentiment is positive,
negative or neutral.  Sentiment analysis can pick up positive or negative feedback,
and give us real-time alerts so that we can easily respond towards any feedback. It’s
also known as opinion mining, generally it emanates the feedback of a customer.
According to our customers positive and negative comments about something
related to our application, we can ensure the good quality of our application by
modifying it. Sentimental analysis can be used in various fields such as Business,
Politics, and Public action and so on. In this project we are mainly focusing on good
quality of applications based on customers review using convolutional neural
network. A convolutional neural network is a class of deep neural networks that have
proven very effective in areas such as image recognition and classification. Here we
are detecting customers good or bad review that can be positive or negative by using
collection of feedback comprise fields such as tourism, costumer review, finance,
software engineering, speech conversation, social media content, news and so on.
. In our proposed model, we train our system to learning the process that identifies
the difference between positive and negative feedback.<br>

The Dataset was collected from <a href="https://www.kaggle.com/lava18/google-play-store-apps">Play store App Review</a>



# importing all the necessary packages
importing all the necessary packages.<br>Pandas is a high performance and easy to use data analysis tool.<br>Numpy for using multi dimensional arrays.<br>NLTK stands for natural language processing.It is a must have tool for language processing.Keras is a deep learning library which makes using deep learing model so easy.<br>Then there is tokenizer that will tokenize every word and pad_sequences is used to ensure that all sequences in a list have the same length.



```python
import pandas as pd
import numpy as np
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
```

    Using TensorFlow backend.
    

# Opening the dataset then Editing

We will convert the csv file as pandas dataframe for easier execution.
dropna() method allows to analyze and drop Rows/Columns with Null values in the dataframe
then we will replace the sentiment labels to numerical value as you know It is easier for computers to read numbers 😁.<br> Our dataset contains three lables.For easier demonstration,we will consider binary classifiaction so we will consider neutarl labelling as positive and hence will give the below 1 for positive and 0 for negative respectively


```python

df = pd.read_csv("googleplaystore_user_reviews.csv", na_values="nan")
df = df.dropna(subset=['App','Translated_Review','Sentiment'], how='any')
df['Sentiment'] = df['Sentiment'].replace(['Positive'],'1')
df['Sentiment'] = df['Sentiment'].replace(['Negative'],'0')
df['Sentiment'] = df['Sentiment'].replace(['Neutral'],'1')

```


# importing some other necessary packages
In this step we will import word_tokenize and stopwords from.Stopword are the words like am,are,i,my,is etc. They are pretty common and does not contain important significance thus they can/should be removed.


```python
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

# reviews are categorized as lines
In this step we will convert the reviews in our dataset to list.We can see we have 37427 reviews.


```python
review_lines = list()
lines = df['Translated_Review'].values.tolist()
print (len(lines))
```

    37427
    

# tokenization and removing punctuation and stop words

We will tokenize the word from every line or reviwe.Then we will do some preprocessing.Well,preprocessing is the process of cleaning or preparing a dataset or set of data for a model.It generally depends on one's personal need as how he/she would preprocess the dataset.But stop_words removing,punctuation removing,numerical value removig are some of the well known preprocessing techniques. Here we will first make all the words lowercase then we will remove punctuation after that we will remove the stop_words. There are also some other well know tecniques for preprocessing like stemming and lemmatization.You can give them a try.<br>
<b>Remember</b> preprocessing is a very necessary steps for any machine learning model.There is a saying "garbage in,garbage out".Your input will be the key to get the desired output.


```python
for line in lines :
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table =str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)
len(review_lines)
#print(review_lines)
    
```




    37427



# word2vec model

Now comes the most important part. Word2Vec is one of the most popular technique to learn word embeddings.Now what is word embedding???<br>

Well,Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.<br><br>

See here <a href="https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba">Word2Vec</a> for more info...<br><br>

We will import Word2vec model from genism.Then we will initialize the model.<br>
We will pass our list (review lines ).Word vector Dimensionality has been given as 100.One can use any dimension s/he needs.But 100 recommended.Window size has been given as 5 which means Maximum distance between the current and predicted word within a sentence should be 5. Workers define how many threads will be used.Generally,Training will be fastrer with multicore machines.

For more information.... <a href ="https://radimrehurek.com/gensim/models/word2vec.html">Genism Word2vec</a>

Then we will print how many words we have got in our vocabulary


```python
import gensim

model = gensim.models.Word2Vec(sentences=review_lines,size=100,window = 5,workers =4,min_count=1)
words = list(model.wv.vocab)
print('total word: %d' %len(words))
```

   E:pranab\New folder\envs\awesome\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
  warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

    total word: 21481
    

# saving the model

We will save the model for later use.We do not need to train word embeddings.By saving we can easily use the model later


```python
filename = 'r.txt'
model.wv.save_word2vec_format(filename,binary=False)


```
E:pranab\New folder\envs\awesome\lib\site-packages\smart_open\smart_open_lib.py:398: UserWarning: This function is deprecated, use smart_open.open instead. See the migration notes for details: https://github.com/RaRe-Technologies/smart_open/blob/master/README.rst#migrating-to-the-new-open-function
  'See the migration notes for details: %s' % _MIGRATION_NOTES_URL
# Mapping words and their corrosponding vectors

In this step first an empty dictionary is initialized then the word2vec model is read.For each word their corresponding vector is mapped from word2Vec model 





```python
import os
embeddings_index = {}
f = open(os.path.join('','r.txt'),encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()

```

# Applying tokenizer 

Tokenizer and pad_sequence is applied on the review sentences.


```python
tk = Tokenizer()
tk.fit_on_texts(review_lines)  #list of texts to train on
sequences = tk.texts_to_sequences(review_lines)  # list of texts to turn to sequences.
word_index = tk.word_index
print("found %s unique tokens " % len(word_index))
review_pad = pad_sequences(sequences,maxlen=100)  #ensureing all sequences in a list to have the same length.
sentiment = df['Sentiment'].values  
print('Shape of review ', review_pad.shape)
print('shape of senti' , sentiment.shape)
```

    found 21481 unique tokens 
    Shape of review  (37427, 100)
    shape of senti (37427,)
    

# map word embeddings from the loaded word2vec model for each word in our word_index items


First, total number of word is initialized.It has to be (1+total word) as it starts from zero index.Then using numpy a (21482*100) size matrix full of zeroes has been created. And for every word their correspondind vector value is matched



```python
num_words = len(word_index) + 1
embedd = np.zeros((num_words,100))

for word , i in word_index.items():
    if i > num_words:
        continue
    embedd_vec = embeddings_index.get(word)
    if embedd_vec is not None:
        embedd[i] = embedd_vec     
print(num_words)
```

    21482
    

# Alexnet model 

In this step,We used AlexNet architecture to build the Sentimental analysis model. The AlexNet architecture consists of five convolutional layers, some of which are followed by maximum pooling layers and then one fully-connected layers and finally store it in sigmoid function..<br>

For output layer Dense layer has been used and units has been set to 1 as our model is to predict binary classification.Sigmoid is used as an activation function.<br>
for calculating loss 'binary_crossentropy' has been used. It's a method of evaluating how well specific algorithm models the given data.There are some other loss function as well.But for binary classifiaction 'binary_crossentropy' is the suitable one.<br>
Adam has been used as the optimizer.The function of optimizer is to minimize the loss.There are also so many optimizers available.You can use any to see which fits well.<br>




```python
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
print("training CNN ...")
model = Sequential()
model.add(Embedding(num_words, embed_dim,
          weights=[embedd], input_length=100, trainable=False))
model.add(Conv1D(num_filters, 7, activation='sigmoid', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='sigmoid', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='sigmoid', padding='same'))
model.add(Conv1D(num_filters, 7, activation='sigmoid', padding='same'))
model.add(Conv1D(num_filters, 7, activation='sigmoid', padding='same'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

```
```python
training CNN ...
_________________________________________________________________
Layer (type)                 Output Shape              Param #    
=================================================================
embedding_9 (Embedding)      (None, 100, 100)          2148200   
_________________________________________________________________
conv1d_39 (Conv1D)           (None, 100, 64)           44864     
_________________________________________________________________
max_pooling1d_21 (MaxPooling (None, 50, 64)            0         
_________________________________________________________________
conv1d_40 (Conv1D)           (None, 50, 64)            28736     
_________________________________________________________________
max_pooling1d_22 (MaxPooling (None, 25, 64)            0         
_________________________________________________________________
conv1d_41 (Conv1D)           (None, 25, 64)            28736     
_________________________________________________________________
conv1d_42 (Conv1D)           (None, 25, 64)            28736     
_________________________________________________________________
conv1d_43 (Conv1D)           (None, 25, 64)            28736     
_________________________________________________________________
max_pooling1d_23 (MaxPooling (None, 12, 64)            0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 768)               0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 768)               0         
_________________________________________________________________
dense_20 (Dense)             (None, 1)                 769       
=================================================================
Total params: 2,308,777
Trainable params: 160,577
Non-trainable params: 2,148,200
```

# Creating Training and Testing set

We will split our dataset into two set.One for trainig and other for testing or validation.20% has been randomly selceted for validation


```python

VALIDATION_SPLIT = 0.2

indices = np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad = review_pad[indices]
sentiment = sentiment[indices]
num_validation = int (VALIDATION_SPLIT * review_pad.shape[0])

X_train_pad = review_pad[:-num_validation]
y_train = sentiment[:-num_validation]
X_test_pad = review_pad[-num_validation:]
y_test = sentiment[-num_validation:]



print('shape of X_train_pad ', X_train_pad.shape)
print('shape of y_train ', y_train.shape)

print('shape of X_test_pad ', X_test_pad.shape)
print('shape of y_test ', y_test.shape)

```

    shape of X_train_pad  (29942, 100)
    shape of y_train  (29942,)
    shape of X_test_pad  (7485, 100)
    shape of y_test  (7485,)
    

# training the classification model on train set and validating on validation set

10 epochs has been given for demonstration purpose


```python
model.fit(X_train_pad,y_train,batch_size=64,epochs=10,validation_data= (X_test_pad,y_test),verbose=2)


```
Train on 29942 samples, validate on 7485 samples
Epoch 1/10
 - 36s - loss: 0.1657 - acc: 0.7771 - val_loss: 0.1454 - val_acc: 0.7848
Epoch 2/10
 - 34s - loss: 0.1384 - acc: 0.7952 - val_loss: 0.1251 - val_acc: 0.8138
Epoch 3/10
 - 34s - loss: 0.1260 - acc: 0.8206 - val_loss: 0.1171 - val_acc: 0.8322
Epoch 4/10
 - 35s - loss: 0.1194 - acc: 0.8306 - val_loss: 0.1146 - val_acc: 0.8366
Epoch 5/10
 - 35s - loss: 0.1147 - acc: 0.8384 - val_loss: 0.1135 - val_acc: 0.8362
Epoch 6/10
 - 34s - loss: 0.1108 - acc: 0.8446 - val_loss: 0.1082 - val_acc: 0.8481
Epoch 7/10
 - 34s - loss: 0.1084 - acc: 0.8476 - val_loss: 0.1084 - val_acc: 0.8470
Epoch 8/10
 - 35s - loss: 0.1034 - acc: 0.8584 - val_loss: 0.1179 - val_acc: 0.8305
Epoch 9/10
 - 34s - loss: 0.1017 - acc: 0.8590 - val_loss: 0.1030 - val_acc: 0.8564
Epoch 10/10
 - 34s - loss: 0.0992 - acc: 0.8641 - val_loss: 0.1025 - val_acc: 0.8569
  


# Evaluting the model


```python
score = model.evaluate(X_test_pad, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))
```

    Testing Accuracy:  0.8569
    

# printing accuracy


```python
print("Accuracy: %.2f%%" % (score[1]*100))
```

    Accuracy: 85.69%
    

# Testing sample dataset

Here we will give there sample test dataset to see whether our model can predict the label of them.If the predicted value is closer to 1 then the review or comment will be positive,if the predicted value is closer to 0 then it will be a negative review


```python
test_sample1="just loving it"
test_sample2="no comments"
test_sample3="totally bad "


test_samples = [test_sample1,test_sample2,test_sample3]
test_samples_tokens = tk.texts_to_sequences(test_samples)

pad =pad_sequences(test_samples_tokens,maxlen=100)

model.predict(x =pad)

```




   array([[0.9669348],
       [0.6705882],
       [0.1296682]], dtype=float32)



# Here we can see that our model has given a good result as it can classify the reviews as positive and negative.

## updated by @adnanemonn from @mrinmoy09

```python

```
