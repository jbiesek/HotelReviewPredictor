import csv
import spacy
from matplotlib import pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob
import numpy as np

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

with open("test.csv", "r") as csvfile:
    test_row_count = sum(1 for line in csvfile) - 1
    print("Test set length:", test_row_count)

with open("train.csv", "r") as csvfile:
    train_row_count = sum(1 for line in csvfile)-1
    print("Train set length:", train_row_count)

def test():
    #https://pypi.org/project/spacytextblob/
    #https://spacy.io/universe/project/spacy-textblob
    i = 0
    with open("test.csv", "r") as csvfile:
        datareader = csv.reader(csvfile)
        x = np.zeros(5)
        y = np.arange(1,6)
        for row in datareader:
            if row[0] == 'Review':
                continue
            review = nlp(row[0])
            # print("Polarity:",review._.blob.polarity)
            polarity = review._.blob.polarity
            # print("Subjectivity: ", review._.blob.subjectivity)
            subjectivity = review._.blob.subjectivity
            # print(review._.blob.sentiment_assessments.assessments)
            sentiment_list = review._.blob.sentiment_assessments.assessments
            postive = 0
            neutral = 0
            negative = 0
            for tuple in sentiment_list:
                if tuple[1] > 0:
                    postive += 1
                elif tuple[1]:
                    negative += 1
                else:
                    neutral += 1
            x[int(row[1])-1] += 1
            # print("Positive:", postive)
            # print("Neutral:", neutral)
            # print("Negative:", negative)
            # print("Rating:",row[1])
            # print("-----------------")
            i += 1
            if i % 100 == 0:
                print(round(((i/test_row_count)*100),2),"%")
    fig = plt.figure()
    plt.bar(y,x)
    plt.title("Test set ratings")
    plt.show()
    fig.savefig("test.png")
    plt.close()


def train():
    # https://pypi.org/project/spacytextblob/
    # https://spacy.io/universe/project/spacy-textblob
    i = 0
    with open("train.csv", "r") as csvfile:
        datareader = csv.reader(csvfile)
        x = np.zeros(5)
        y = np.arange(1,6)
        for row in datareader:
            if row[0] == 'Review':
                continue
            review = nlp(row[0])
            # print("Polarity:",review._.blob.polarity)
            polarity = review._.blob.polarity
            # print("Subjectivity: ", review._.blob.subjectivity)
            subjectivity = review._.blob.subjectivity
            # print(review._.blob.sentiment_assessments.assessments)
            sentiment_list = review._.blob.sentiment_assessments.assessments
            postive = 0
            neutral = 0
            negative = 0
            for tuple in sentiment_list:
                if tuple[1] > 0:
                    postive += 1
                elif tuple[1]:
                    negative += 1
                else:
                    neutral += 1
            x[int(row[1])-1] += 1
            # print("Positive:", postive)
            # print("Neutral:", neutral)
            # print("Negative:", negative)
            # print("Rating:",row[1])
            # print("-----------------")
            i += 1
            if i % 100 == 0:
                print(round(((i / train_row_count) * 100), 2), "%")
    fig = plt.figure()
    plt.bar(y, x)
    plt.title("Train set ratings")
    plt.show()
    fig.savefig("train.png")
    plt.close()

test()
train()
