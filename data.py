import csv
import os.path

from sklearn.model_selection import train_test_split
import stanza
import string
import numpy as np


class Data:
    x = []
    y = []
    x_test = []
    id_test = []
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    bool_valid = True
    stop_words = ["այդ", "այլ", "այն", "այս", "դու", "դուք", "եմ", "են", "ենք", "ես", "եք", "է", "էի", "էին", "էինք",
                  "էիր", "էիք", "էր",
                  "ըստ", "թ", "ի", "ին", "իսկ", "իր", "կամ", "համար", "հետ", "հետո", "մենք", "մեջ", "մի", "ն", "նա",
                  "նաև", "նրա", "նրանք", "որ", "որը",
                  "որոնք", "որպես", "ու", "ում", "պիտի", "վրա", "և"]
    punctuation = list(string.punctuation)

    def extract(self, valid):
        self.bool_valid = valid
        with open("dataset/train.csv", "r") as train_file:
            reader = csv.reader(train_file)

            for elem in reader:
                if elem[0] == "Id":
                    continue
                self.y.append(int(elem[2]))

        with open("dataset/lemma.csv", "r") as train_file:
            reader = csv.reader(train_file)
            for elem in reader:
                if elem[0] == "Id":
                    continue
                self.x.append(elem[1])

        self.split()

    def split(self):
        with open("dataset/test_lemma.csv", "r") as train_file:
            reader = csv.reader(train_file)
            for elem in reader:
                if elem[0] == "Id":
                    continue
                self.id_test.append(elem[0])
                self.x_test.append(elem[1])

        # stanza.download("hy")
        # nlp = stanza.Pipeline("hy", use_gpu=False)
        # test_lemma_x = []
        # for text in self.x_test:
        #     str = ""
        #     doc = nlp(text)
        #     for sent in doc.sentences:
        #         for word in sent.words:
        #             if word.lemma not in self.punctuation:
        #                 str += word.lemma + " "
        #     test_lemma_x.append(str)

        # with open(os.path.join("dataset", "test_lemma.csv"), "w") as file:
        #     writer = csv.writer(file)
        #     text = ["Id", "Text"]
        #     writer.writerow(text)
        #     for i in range(0, len(test_lemma_x)):
        #         writer.writerow([i, test_lemma_x[i]])

        self.x_train = self.x
        self.y_train = self.y
        values, counts = np.unique(self.y, return_counts=True)
        print(counts)
        if self.bool_valid:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x, self.y, test_size=0.2)

    @staticmethod
    def save_to_csv(result, test_id, file_name):
        with open(os.path.join("dataset", file_name), "w") as file:
            writer = csv.writer(file)
            text = ["Id", "Category"]
            writer.writerow(text)
            for i in range(0, len(result)):
                writer.writerow([test_id[i], result[i]])
