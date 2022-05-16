import csv
from data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class TfIdfVectorizerTrain:
    tfidf_vectorizer = TfidfVectorizer(min_df=10, max_df=0.8)
    neuro = None
    result = None
    data = None

    def train(self, data):
        self.data = data
        features = self.tfidf_vectorizer.fit_transform(self.data.x_train)
        self.neuro = Sequential()
        self.neuro.add(Dense(1000, input_dim=features.shape[1], activation="relu"))
        self.neuro.add(Dense(7, activation="sigmoid"))
        self.neuro.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        self.neuro.fit(features.toarray(), np.array(self.data.y_train), epochs=5, batch_size=10)
        valid_features = self.tfidf_vectorizer.transform(self.data.x_valid).toarray()
        score = self.neuro.evaluate(valid_features, np.array(self.data.y_valid))
        return score

    def predict(self):
        test_features = self.tfidf_vectorizer.transform(self.data.x_test).toarray()
        self.result = self.neuro.predict(test_features)
        self.result = self.result.argmax(-1)
        Data.save_to_csv(self.result, self.data.id_test, "tfidf.csv")

