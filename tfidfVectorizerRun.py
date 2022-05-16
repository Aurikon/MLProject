from TfidfVectorizerTrain import TfIdfVectorizerTrain
from data import Data


def main():
    data = Data()
    data.extract()
    tfidf = TfIdfVectorizerTrain()
    score = tfidf.train(data)
    tfidf.predict()
    print(score)


if __name__ == "__main__":
    main()
