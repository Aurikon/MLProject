from data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class TfIdfVectorizerTrain:
    tfidf_vectorizer = TfidfVectorizer(stop_words=Data.stop_words)
    neuro = None
    result = None
    data = None

    def train(self, data, epoch_count):
        self.data = data
        score = 0
        features = self.tfidf_vectorizer.fit_transform(self.data.x_train)
        # train_x = MaxAbsScaler().fit_transform(features)
        print("input neuros", features.shape[1])
        self.neuro = Sequential()
        self.neuro.add(Dense(1000, input_dim=features.shape[1], activation="relu"))
        # self.neuro.add(Dense(int(features.shape[1] / 4), activation="relu"))
        # self.neuro.add(Dense(int(features.shape[1] / 6), activation="relu"))
        # self.neuro.add(Dense(int(features.shape[1] / 8), activation="relu"))
        # self.neuro.add(Dense(int(features.shape[1] / 10), activation="relu"))
        # self.neuro.add(Dense(50, activation="relu"))
        self.neuro.add(Dense(7, activation="sigmoid"))
        self.neuro.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.neuro.fit(features.toarray(), np.array(self.data.y_train), epochs=epoch_count)
        # valid_features = self.tfidf_vectorizer.transform(self.data.x_valid).toarray()
        # score = self.neuro.evaluate(valid_features, np.array(self.data.y_valid))
        return score

    def predict(self, filename):
        test_features = self.tfidf_vectorizer.transform(self.data.x_test).toarray()
        self.result = self.neuro.predict(test_features)
        self.result = self.result.argmax(-1)
        Data.save_to_csv(self.result, self.data.id_test, filename)

