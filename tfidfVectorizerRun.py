from TfidfVectorizerTrain import TfIdfVectorizerTrain
from data import Data


def main():
    data = Data()
    data.extract()
    tfidf = TfIdfVectorizerTrain()

    score = tfidf.train(data, 6)
    print(score)

    tfidf.predict("tflemma6epochstopwords.csv")


if __name__ == "__main__":
    main()
