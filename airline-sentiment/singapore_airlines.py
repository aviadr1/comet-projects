# Comet
from comet_ml import Experiment

# Standard packages
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# nltk for preprocessing of text data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# sklearn for preprocessing and machine learning models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# XGBoost for Machine Learning (Gradient Boosting Machine (GBM))
import xgboost as xgb

# Keras for neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Random seeds for consistent results
from tensorflow import set_random_seed
seed = 1234
np.random.seed(seed)
set_random_seed(seed)

class PreProcessor:
    '''
    Easily performs all the standard preprocessing steps
    like removing stopwords, stemming, etc.
    Only input that you need to provide is the dataframe and column name for the tweets
    '''
    def __init__(self, df, column_name):
        self.data = df
        self.conversations = list(self.data[column_name])
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.preprocessed = []

    def tokenize(self, sentence):
        '''
        Splits up words and makes a list of all words in the tweet
        '''
        tokenized_sentence = word_tokenize(sentence)
        return tokenized_sentence

    def remove_stopwords(self, sentence):
        '''Removes stopwords like 'a', 'the', 'and', etc.'''
        filtered_sentence = []
        for w in sentence:
            if w not in self.stopwords and len(w) > 1 and w[:2] != '//' and w != 'https':
                filtered_sentence.append(w)
        return filtered_sentence

    def stem(self, sentence):
        '''
        Stems certain words to their root form.
        For example, words like 'computer', 'computation'
        all get truncated to 'comput'
        '''
        return [self.stemmer.stem(word) for word in sentence]

    def join_to_string(self, sentence):
        '''
        Joins the tokenized words to one string.
        '''
        return ' '.join(sentence)

    def full_preprocess(self, n_rows=None):
        '''
        Preprocess a selected number of rows and
        connects them back to strings
        '''
        # If nothing is given do it for the whole dataset
        if n_rows == None:
            n_rows = len(self.data)

        # Perform preprocessing
        for i in range(n_rows):
            tweet = self.conversations[i]
            tokenized = self.tokenize(tweet)
            cleaned = self.remove_stopwords(tokenized)
            stemmed = self.stem(cleaned)
            joined = self.join_to_string(stemmed)
            self.preprocessed.append(joined)
        return self.preprocessed



DATA = '/Users/nikolaskaris/Desktop/twitter-airline-sentiment/'


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

def main():

    experiment = Experiment(api_key="kZ9EvZnOe5Kdw40yNNY4UXNu3",
                        project_name="singapore-airlines", workspace="demos")

    experiment.log_metric('val_acc', 6)
    experiment.log_hyperparamter('test_parameter', 'hello')
    experiment.log_dataset_hash(x_train)

    raw_df = pd.read_csv(f'{DATA}Tweets.csv')

    df = raw_df[['tweet_id', 'text', 'airline_sentiment']]

    # Preprocess text and put it in a new column
    preprocessor = PreProcessor(df, 'text')
    df['cleaned_text'] = preprocessor.full_preprocess()

    # Shuffling so we can get random tweets for the test set
    df = shuffle(df, random_state=seed)
    # Keep 1000 samples of the data as test set
    test_set = df[:1000]

    # Get training and validation data
    X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'][1000:],
                                                      df['airline_sentiment'][1000:],
                                                      test_size=0.2,
                                                      random_state=seed)

    # Get sentiment labels for test set
    y_test = test_set['airline_sentiment']

    # Create matrix based on word frequency in tweets
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(test_set['cleaned_text'])

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
    optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
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
    epochs = 50
    batch_size = 32

    # Fit the model using the batch_generator
    hist = model.fit_generator(generator=batch_generator(X_train, onehot_encoded_train, batch_size=batch_size, shuffle=True),
                               epochs=epochs, validation_data=(X_val, onehot_encoded_val),
                               steps_per_epoch=X_train.shape[0]/batch_size, callbacks=[es])


if __name__ == "__main__":
    main()
