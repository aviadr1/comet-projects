# Getting Started with Natural Language Processing: US Airline Tweet Sentiment Analysis
![](https://www.comet.ml/api/image/notes/download?imageId=KfIlDZ4EkR3ofDsEIM8moiq5j&objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

# Comet
To use this getting started project to start logging NLP modeling tasks to [Comet.ml](www.comet.ml), make sure you have signed up and know your Comet API_KEY. Please consult the [docs](https://www.comet.ml/docs/) for more information and setting up your Comet configuration before running experiments.mv 

#Sections

1. Introduction to NLP
2. Dataset Exploration
3. NLP Preprocessing 
4. Train Models
5. Hyperparameter Optimization
6. Where to go from here / Future work

## Introduction to NLP

Natural Language Processing (NLP) is a subfield of machine learning concerned with processing and analyzing large amounts of natural language data, usually in the form of text or audio. Some common challenges within NLP include speech recognition, text generation, and sentiment analysis, while some high-profile products deploying NLP models include Apple's Siri, Amazon's Alexa, and many of the chatbots one might interact with online. 

To get started with NLP and introduce some of the core concepts in the field, we're going to build a model that tries to predict the sentiment (positive, neutral, or negative) of tweets relating to US Airlines, using the popular [Twitter US Airline Sentiment dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).

Code snippets will be included in this post, but for fully reproducible notebooks and scripts, navigate to the `Files` tab at the top of this page. 

## Dataset Exploration

Let's start by importing some libraries. Make sure to install:

`comet_ml` for experiment management, visualizations, code tracking and hyperparameter optimization

```
# Comet
import comet_ml
from comet_ml import Experiment 
```

Some standard packages, like pandas, numpy and matplotlib

```
# Standard packages
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

`nltk` for some key natural language processing functions

```
# nltk for preprocessing of text data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
```

`sklearn` and `keras` for machine learning models

```
# sklearn for preprocessing and machine learning models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras for neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
```

Load the data

`raw_df = pd.read_csv('.../twitter-airline-sentiment/Tweets.csv'}`

Let's check the shape of the dataframe.

`raw_df.shape()`
`
```
(14640, 15)
```

So we've got 14,640 samples (tweets), each with 15 features. Let's take a look at what features this dataset contains.

`raw_df.columns`
```
Index(['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'text', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone'],
      dtype='object')
```

Let's also take a look at airline sentiment for each airline.

![](https://www.comet.ml/api/image/notes/download?imageId=6zhxiY28dmcoMsmWeoUQgmitt&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Link to Comet experiment code](https://www.comet.ml/demo/nlp-airline/99bcfee71c74405c84d2da1766ee4374?experiment-tab=code)

![](https://www.comet.ml/api/image/notes/download?imageId=JD8TXrnwmlMuwx6ogZcgEJB3S&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

![](https://www.comet.ml/api/image/notes/download?imageId=lrrMI6qq3CWySJ10BajdGOr6R&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

![](https://www.comet.ml/api/image/notes/download?imageId=tuFMKl06OhURlMwpdAL8Nn53R&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

![](https://www.comet.ml/api/image/notes/download?imageId=1yWbqN6GtcvAoAZSOEWK0h6kb&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

![](https://www.comet.ml/api/image/notes/download?imageId=OrcbtV7xtXG0GPcYWt22z40cJ&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

![](https://www.comet.ml/api/image/notes/download?imageId=Mu4TbEyW18RwnYEYrQEG3A02y&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Link to Comet experiment](https://www.comet.ml/demo/nlp-airline/notes)

Every airline has more negative tweets than either neutral or positive tweets, with Virgin America receiving the most balanced spread of positive, neutral and negative of all the US airlines. While we're going to focus on NLP-specific analysis in this write-up, there are excellent sources of further feature-engineering and exploratory data analysis. Kaggle kernels [here](https://www.kaggle.com/parthsharma5795/comprehensive-twitter-airline-sentiment-analysis) and [here](https://www.kaggle.com/mrisdal/exploring-audience-text-length) are particularly instructive in analyzing features such as audience and tweet length as related to sentiment. 

Let's create a new dataframe with only `tweet_id`, `text` and `airline_sentiment` features. 

```
df = raw_df[['tweet_id', 'text', 'airline_sentiment']]
```

And now let's take a look at a few of the tweets themselves. What's the data look like?

![](https://www.comet.ml/api/image/notes/download?imageId=PjcMv2dYZz19mpDEep3uLG1Zb&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

Next, we're going to conduct a few standard NLP preprocessing techniques to get our dataset ready for training. 

## NLP: Text Preprocessing

For the purposes of constructing NLP models, one must conduct some basic steps of text preprocessing in order to transfer text from human language to machine readable format for further processing. Here we will cover some of the standard practices: *tokenization, stopword removal, and stemming*. You can consult [this post](https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908) to learn about additional text preprocessing techniques.


### Tokenization

Given a character sequence and a defined document unit, tokenization is the task of chopping it up into discrete pieces called *tokens*. In the process of chopping up text, tokenization also commonly involves throwing away certain characters, such as punctuation. 

![](https://www.comet.ml/api/image/notes/download?imageId=hj2n9zEDdMQZSNlyreG93UiBf&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

It is simple (and often useful) to think of tokens simply as words, but to fine tune your understanding of the specific terminology of NLP tokenization, the [Stanford NLP group's overview](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) is quite useful. 

The NLTK library has a built-in [tokenizer](https://www.nltk.org/api/nltk.tokenize.html) we will use to tokenize the US Airline Tweets. 

```
from nltk.tokenize import word_tokenize

def tokenize(sentence):
    tokenized_sentence = word_tokenize(sentence)
    return tokenized_sentence
```

### Stop word Removal

Sometimes, common words that may be of little value in determing the semantic quality of a document are excluded entirely from the vocabulary. These are called *stop words*. A general strategy for determining a list of stop words is to sort the terms by *collection frequency* (total number of times each term appears in the document) and then to filter out the most frequent terms as a stop list — hand-filtered by semantic content.

![](https://www.comet.ml/api/image/notes/download?imageId=gO0Zdrexgi3mcUVDxv6WJC0JN&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Source](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html): Stanford NLP group

The NLTK library has a standard stopword list we will adopt here. 
```
from nltk.corpus import stopwords
class PreProcessor:
    '''
    def __init__(self, df, column_name):
        '''
        self.stopwords = set(stopwords.words('english'))

    def remove_stopwords(self, sentence):
        filtered_sentence = []
        for w in sentence:
            if w not in self.stopwords and len(w) &gt; 1 and w[:2] != '//' and w != 'https':
                filtered_sentence.append(w)
        return filtered_sentence 
```


### Stemming

For grammatical purposes, documents use different forms of a word (look, looks, looking, looked) that in many situations have very similar semantic qualities. Stemming is a rough process by which variants or related forms of a word are reduced (stemmed) to a common base form. As stemming is a removal of prefixed or suffixed letters from a word, the output may or may not be a word belonging to the language corpus. *Lemmatization* is a more precise process by which words are properly reduced to the base word from which they came. 

Examples:

*Stemming*: car, cars, car's, cars' *become* car

*Lemmatization*: am, are is *become* be

*Stemmed and Lemmatized Sentence*: 'the boy's cars are different colors' *become* 'the boy car is differ color'

The most common algorithm for stemming English text is [Porter's algorithm](TO DO). [Snowball](http://snowball.tartarus.org/texts/introduction.html), a language for stemming algorithms, was developed by Porter in 2001 and is the basis for the NLTK implementation of its SnowballStemmer, which we will use here.

```
from nltk.stem.snowball import SnowballStemmer
class PreProcessor:
    '''
    def __init__(self, df, column_name):
        '''
        self.stemmer = SnowballStemmer(&quot;english&quot;)

    def stem(self, sentence):
        return [self.stemmer.stem(word) for word in sentence]

```

Code for these preprocessing steps can be found [here](LINK GITHUB REPO).

Next we'll create a PreProcessor and run it on the 'text' column of our dataframe to tokenize, stem, and remove stopwords from the tweets.

```
preprocessor = PreProcessor(df, 'text')
df['cleaned_text'] = preprocessor.full_preprocess()
```

```
df = shuffle(df, random_state=seed)
# Create test set
test_set = df[:1000]

# Create train and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'][1000:],
                                                      df['airline_sentiment'][1000:],
                                                      test_size=0.2,
                                                      random_state=seed)

# Get sentiment labels for test set
y_test = test_set['airline_sentiment']
```

Now that we've split our data into train, validation and test sets, we'll TF-IDF vectorize them

*TF-IDF Vectorization*

TFIDF, or *term frequency - inverse document frequency*, is a numerical statitistic that reflects how important a word is to a document in a collection or corpus. It is often used to produce weights associated with words that can be useful in searches of information retrieval or text mining. The tf-idf value of a word increases proportionally to the number of times a word appears in a document, and is offset by the number of documents in the corpus that contain that word. This offset helps adjust for the fact that some words appear more frequently in general (think of how stopwords like 'a', 'the', 'to' might have incredibly high tf-idf values if not for offsetting). 

![](https://www.comet.ml/api/image/notes/download?imageId=5auX7Gg6bvNaN7wjZVx4fLZXi&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)
[**Source**](https://becominghuman.ai/word-vectorizing-and-statistical-meaning-of-tf-idf-d45f3142be63)

We will use scikit-learn's implementation of [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), which converts a collection of raw documents (our twitter dataset) into a matrix of TF-IDF features.

```
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(test_set['cleaned_text'])
```

###Training

We are ready to start training our model. The first thing we'll do is create a Comet experiment object:

```experiment = Experiment(api_key = 'your-personal-key', project_name = 'nlp-airline', workspace = 'demo')```

Next, we'll build an [Light Gradient-Boosting classifier (LGBM)](https://lightgbm.readthedocs.io/en/latest/), an [XGBoost classifier](https://xgboost.readthedocs.io/en/latest/), and a relatively straightforward [neural network with keras](https://keras.io/models/sequential/) and compare how each of these models performs. Oftentimes it's hard to tell which architecture will perform best without testing them out. Comet's project-level view helps make it easy to compare how different experiments are performing and let you easily move from model selection to model tuning.

#### LGBM
```
# sklearn's Gradient Boosting Classifier (GBM)
gbm = GradientBoostingClassifier(n_estimators=200, 
                                 max_depth=6, 
                                 random_state=seed)
gbm.fit(X_train, y_train)
# Check results
train_pred = gbm.predict(X_train)
val_pred = gbm.predict(X_val)
print(f'Accuracy on training set (GBM): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (GBM): {round(accuracy_score(y_val,val_pred)*100, 4)}%')

val_accuracy = round(accuracy_score(y_val,val_pred)*100, 4)
train_accuracy = round(accuracy_score(y_train, train_pred)*100, 4)

# log to comet
experiment.log_metric('val_acc', val_accuracy)
experiment.log_metric('Accuracy', train_accuracy)
```

#### XGBOOST


```
xgb_params = {'objective' : 'multi:softmax',
              'eval_metric' : 'mlogloss',
              'eta' : 0.1,
              'max_depth' : 6,
              'num_class' : 3,
              'lambda' : 0.8,
              'estimators' : 200,
              'seed' : seed
              
}

# Transform categories into numbers
# negative = 0, neutral = 1 and positive = 2
target_train = y_train.astype('category').cat.codes
target_val = y_val.astype('category').cat.codes

# Transform data into a matrix so that we can use XGBoost
d_train = xgb.DMatrix(X_train, label = target_train)
d_val = xgb.DMatrix(X_val, label = target_val)

# Fit XGBoost
watchlist = [(d_train, 'train'), (d_val, 'validation')]
bst = xgb.train(xgb_params, 
                d_train, 
                400,  
                watchlist,
                early_stopping_rounds = 50, 
                verbose_eval = 0)

# Check results for XGBoost
train_pred = bst.predict(d_train)
val_pred = bst.predict(d_val)
print(f'Accuracy on training set (XGBoost): {round(accuracy_score(target_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (XGBoost): {round(accuracy_score(target_val, val_pred)*100, 4)}%')
```

#### Neural Net

```
# Generator so we can easily feed batches of data to the neural network
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

# Onehot encoding of target variable
# Negative = [1,0,0], Neutral = [0,1,0], Positive = [0,0,1]

# Initialize sklearn's one-hot encoder class
onehot_encoder = OneHotEncoder(sparse=False)

# One hot encoding for training set
integer_encoded_train = np.array(y_train).reshape(len(y_train), 1)
onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)

# One hot encoding for validation set
integer_encoded_val = np.array(y_val).reshape(len(y_val), 1)
onehot_encoded_val = onehot_encoder.fit_transform(integer_encoded_val)

# Neural network architecture
initializer = keras.initializers.he_normal(seed=seed)
activation = keras.activations.elu
optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)

# Build model architecture
model = Sequential()
model.add(Dense(20, activation=activation, kernel_initializer=initializer, input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax', kernel_initializer=initializer))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Hyperparameters
epochs = 15
batch_size = 16

# Fit the model using the batch_generator
hist = model.fit_generator(generator=batch_generator(X_train, onehot_encoded_train, batch_size=batch_size, shuffle=True),
                           epochs=epochs, validation_data=(X_val, onehot_encoded_val),
                           steps_per_epoch=X_train.shape[0]/batch_size, callbacks=[es])
```

Comparing our models using Comet's project view, we can see that our Neural Network models are outperforming the XGBoost and LGBM experiments by a considerable margin.

![](https://www.comet.ml/api/image/notes/download?imageId=m8fnRP8X3QsenAuDKxmoB1YBU&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Comet Project](https://www.comet.ml/demo/nlp-airline/view/j1ZRx1zuXUmju7PBvRKlZEzlV)

Let's select the neural net architecture for now and fine tune it. *Note*, since we've stored all of our experiments — including the XGBoost and LGBM runs we're not going to use right now — if we decide we'd like to revisit those architectures in the future, all we'll have to do is view those experiments in the Comet project page and we'll be able to reproduce them instantly.

###Hyperparameter Optimization

Now that we've selected our architecture from an initial search of XGBoost, LGBM and a simple keras-implementation of a neural network, we'll need to conduct a hyperparameter optimization to fine-tune our model. Hyperparameter optimization can be an incredibly difficult, computationally expensive, and slow process for complicating modeling tasks. Comet has built an [optimization service](https://www.comet.ml/docs/python-sdk/introduction-optimizer/) that can conduct this search for you. Simply pass in the algorithm you'd like to sweep the hyperparameter space with, hyperparameters and ranges to search, and a metric to minimize or maximize, and Comet can handle this part of your modeling process for you. 

![](https://www.comet.ml/api/image/notes/download?imageId=9jC2i4hNxU2XmPWosNqRu91qR&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Link to Optimizer code in Comet Experiment](https://www.comet.ml/demo/nlp-airline/258a9e3df84346e3bb503aff758cb134?experiment-tab=code)

After running our optimization, it is straightforward to select the hyperparameter configuration that yielded the highest accuracy, lowest loss, or whatever performance you were seeking to optimize. Here we keep the optimization problem rather simple: we only search ```epoch```, ```batch_size```, and ```dropout```. The parallel coordinates chart shown below, another native Comet feature, provides a useful visualization of the underlying hyperparameter space our Optimizer has traversed:

![](https://www.comet.ml/api/image/notes/download?imageId=TV0ZV4h9vsPrA45gN4Sz3AYqS&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Link to Comet Project](https://www.comet.ml/demo/nlp-airline/view/j1ZRx1zuXUmju7PBvRKlZEzlV)

Let's run another optimization sweep, this time including a range of learning rates to test. 

![](https://www.comet.ml/api/image/notes/download?imageId=sP7w7y0qKj6HR2EmAMBDGlaiA&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Link to Comet Project](https://www.comet.ml/demo/nlp-airline/view/j1ZRx1zuXUmju7PBvRKlZEzlV)

And again we get a view into the regions of the underlying hyperparameter space that are yielding higher `val_acc` values. 

Say now we'd like to compare the performance of two of our better models to keep fine-tuning. Simply select two experiments from your list and click the `Diff` button and Comet will allow you to visually inspect every code and hyperparameter change, as well as side-by-side visualizations of both experiments. 

![](https://www.comet.ml/api/image/notes/download?imageId=4708WXUt3lfEimmXdubPDBd48&amp;objectId=ef4e6b5d1a4e4767b036d797f9b9bded)

[Comet Experiment Diff](https://www.comet.ml/demo/nlp-airline/258a9e3df84346e3bb503aff758cb134/ee2949dac5d74dc789103f03b986ff80/compare)

From here you can continue your model building. Fine tune one of the models we've pulled out of the architecture comparison and parameter optimization sweeps, or go back to the start and compare new architectures against our baseline models. All of your work is saved in your Comet project space. 

# Where To Go From Here / Future Learning

For additional learning resources in NLP, check out fastai's new [NLP course](https://www.fast.ai/2019/07/08/fastai-nlp/) or this [blog post](https://medium.com/huggingface/the-best-and-most-current-of-modern-natural-language-processing-5055f409a1d1) published by Hugging Face that covers some of the best recent papers and trends in NLP. 





