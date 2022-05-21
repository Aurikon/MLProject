import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
import sklearn.metrics as metrics
from data import Data


class TfIdfVectorizerLogistic:
    tfidf_vectorizer = TfidfVectorizer(stop_words=Data.stop_words)
    data = None
    logistic_regression = None
    result = None

    def train(self, data):
        self.data = data
        features = self.tfidf_vectorizer.fit_transform(self.data.x_train)
        train_x = MaxAbsScaler().fit_transform(features)
        # valid_features = self.tfidf_vectorizer.transform(self.data.x_valid)
        # valid_x = MaxAbsScaler().fit_transform(valid_features)

        self.logistic_regression = LogisticRegression(penalty="l1", solver="liblinear").fit(train_x, self.data.y_train)

    def predict(self, filename):
        features = self.tfidf_vectorizer.transform(self.data.x_test)
        test_x = MaxAbsScaler().fit_transform(features)
        self.result = self.logistic_regression.predict(test_x.toarray())
        Data.save_to_csv(self.result, self.data.id_test, filename)


def main():
    data = Data()
    data.extract()
    tf = TfIdfVectorizerLogistic()
    tf.train(data)
    # tf.predict("logistic2.csv")


if __name__ == "__main__":
    main()
