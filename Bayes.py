from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from data import Data


class TfidfBayes:
    bayes_classificator = MultinomialNB(alpha=0.2)
    data = None
    result = None
    tfidf_vectorizer = TfidfVectorizer(stop_words=Data.stop_words)

    def train(self, data):
        self.data = data
        features = self.tfidf_vectorizer.fit_transform(self.data.x_train)
        self.bayes_classificator.fit(features, data.y_train)

        if self.data.bool_valid:
            valid_features = self.tfidf_vectorizer.transform(self.data.x_valid)
            score = self.bayes_classificator.score(valid_features, self.data.y_valid)
            print(score)

    def predict(self, filename):
        features = self.tfidf_vectorizer.transform(self.data.x_test)
        result = self.bayes_classificator.predict(features)
        Data.save_to_csv(result, self.data.id_test, filename)


def main():
    data = Data()
    data.extract(False)
    tf_bayes = TfidfBayes()
    tf_bayes.train(data)
    tf_bayes.predict("tfidfbayes2.csv")


if __name__ == "__main__":
    main()
