import csv
import os.path

from sklearn.model_selection import train_test_split


class Data:
    x = []
    y = []
    x_test = []
    id_test = []
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    def extract(self):
        with open("dataset/train.csv", "r") as train_file:
            reader = csv.reader(train_file)

            for elem in reader:
                if elem[0] == "Id":
                    continue
                self.x.append(elem[1])
                self.y.append(int(elem[2]))

            self.split()

    def split(self):
        with open("dataset/test.csv", "r") as train_file:
            reader = csv.reader(train_file)
            for elem in reader:
                if elem[0] == "Id":
                    continue
                self.id_test.append(elem[0])
                self.x_test.append(elem[1])

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x, self.y, test_size=0.1)

    @staticmethod
    def save_to_csv(result, test_id, file_name):
        with open(os.path.join("dataset", file_name), "w") as file:
            writer = csv.writer(file)
            text = ["Id", "Category"]
            writer.writerow(text)
            for i in range(0, len(result)):
                writer.writerow([test_id[i], result[i]])
