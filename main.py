import csv
import spacy
from matplotlib import pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews?resource=download
with open("test.csv", "r") as csvfile:
    test_row_count = sum(1 for line in csvfile) - 1
    print("Test set length:", test_row_count)

with open("train.csv", "r") as csvfile:
    train_row_count = sum(1 for line in csvfile) - 1
    print("Train set length:", train_row_count)


def test():
    # https://pypi.org/project/spacytextblob/
    # https://spacy.io/universe/project/spacy-textblob
    i = 0
    with open("test.csv", "r") as csvfile:
        with open('test_sentiment.csv', 'w') as f:
            datareader = csv.reader(csvfile)
            writer = csv.writer(f)
            x = np.zeros(5)
            y = np.arange(1, 6)
            writer.writerow(["Polarity", "Subjectivity", "Positive", "Neutral", "Negative", "Rating"])
            for row in datareader:
                if row[0] == 'Review':
                    continue
                review = nlp(row[0])
                polarity = review._.blob.polarity
                subjectivity = review._.blob.subjectivity
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
                x[int(row[1]) - 1] += 1
                data = [polarity, subjectivity, postive, neutral, negative, row[1]]
                writer.writerow(data)
                i += 1
                if i % 100 == 0:
                    print(round(((i / test_row_count) * 100), 2), "%")
    fig = plt.figure()
    plt.bar(y, x)
    plt.title("Test set ratings")
    plt.show()
    fig.savefig("test.png")
    plt.close()


def train():
    # https://pypi.org/project/spacytextblob/
    # https://spacy.io/universe/project/spacy-textblob
    i = 0
    with open("train.csv", "r") as csvfile:
        with open('train_sentiment.csv', 'w') as f:
            datareader = csv.reader(csvfile)
            writer = csv.writer(f)
            x = np.zeros(5)
            y = np.arange(1, 6)
            writer.writerow(["Polarity", "Subjectivity", "Positive", "Neutral", "Negative", "Rating"])
            for row in datareader:
                if row[0] == 'Review':
                    continue
                review = nlp(row[0])
                polarity = review._.blob.polarity
                subjectivity = review._.blob.subjectivity
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
                x[int(row[1]) - 1] += 1
                i += 1
                data = [polarity, subjectivity, postive, neutral, negative, row[1]]
                writer.writerow(data)
                if i % 100 == 0:
                    print(round(((i / train_row_count) * 100), 2), "%")
    fig = plt.figure()
    plt.bar(y, x)
    plt.title("Train set ratings")
    plt.show()
    fig.savefig("train.png")
    plt.close()


def sentiment_analysis(input):
    review = nlp(input)
    polarity = review._.blob.polarity
    subjectivity = review._.blob.subjectivity
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
    df = pd.DataFrame(list(zip([polarity], [subjectivity], [postive], [neutral], [negative])),
                      columns=["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"])
    return df


def linear_regression_predict():
    # https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/

    with open('linear_regression_predictions.txt', 'w') as f:
        train = pd.read_csv("train_sentiment.csv")
        test = pd.read_csv("test_sentiment.csv")
        predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
        X_train = train[predictors]
        y_train = train["Rating"]
        X_test = test[predictors]
        y_test = test["Rating"]
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        prediction = lm.predict(X_test)
        for i in range(test_row_count):
            if prediction[i] > 5:
                prediction[i] = 5
            elif prediction[i] < 1:
                prediction[i] = 1
        avg_mean = 0
        avg_mean_sq = 0
        for i in range(test_row_count):
            mean = abs(prediction[i] - y_test[i])
            mean_sq = (abs(prediction[i] - y_test[i])) ** 2
            print("Predicted rating:", prediction[i], "Actual rating:", y_test[i], "Mean:", mean, "Mean^2:", mean_sq)
            txt = "Predicted rating: " + str(prediction[i]) + " Actual rating: " + str(y_test[i]) + " Mean: " + str(
                mean) + " Mean^2: " + str(mean_sq)
            f.write(txt)
            f.write("\n")
            avg_mean += mean
            avg_mean_sq += mean_sq
        avg_mean = avg_mean / test_row_count
        avg_mean_sq = avg_mean_sq / test_row_count
        print("Average mean:", avg_mean, "Average mean^2:", avg_mean_sq)
        txt = "Average mean: " + str(avg_mean) + " Average mean^2: " + str(avg_mean_sq)
        f.write(txt)


def logistic_regression_predict():
    with open('logistic_regression_predictions.txt', 'w') as f:
        train = pd.read_csv("train_sentiment.csv")
        test = pd.read_csv("test_sentiment.csv")
        predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
        X_train = train[predictors]
        y_train = train["Rating"]
        X_test = test[predictors]
        y_test = test["Rating"]
        lm = LogisticRegression(
            solver='newton-cg')  # possible solvers: 'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'
        lm.fit(X_train, y_train)
        prediction = lm.predict(X_test)
        avg_mean = 0
        avg_mean_sq = 0
        for i in range(test_row_count):
            mean = abs(prediction[i] - y_test[i])
            mean_sq = (abs(prediction[i] - y_test[i])) ** 2
            print("Predicted rating:", prediction[i], "Actual rating:", y_test[i], "Mean:", mean, "Mean^2:", mean_sq)
            txt = "Predicted rating: " + str(prediction[i]) + " Actual rating: " + str(y_test[i]) + " Mean: " + str(
                mean) + " Mean^2: " + str(mean_sq)
            f.write(txt)
            f.write("\n")
            avg_mean += mean
            avg_mean_sq += mean_sq
        avg_mean = avg_mean / test_row_count
        avg_mean_sq = avg_mean_sq / test_row_count
        print("Average mean:", avg_mean, "Average mean^2:", avg_mean_sq)
        txt = "Average mean: " + str(avg_mean) + " Average mean^2: " + str(avg_mean_sq)
        f.write(txt)

def polynomial_regression_predict():
    # https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386

    with open('polynomial_regression_predictions.txt', 'w') as f:
        train = pd.read_csv("train_sentiment.csv")
        test = pd.read_csv("test_sentiment.csv")
        predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
        X_train = train[predictors]
        y_train = train["Rating"]
        X_test = test[predictors]
        y_test = test["Rating"]
        poly_reg = PolynomialFeatures(degree=5)
        X_poly = poly_reg.fit_transform(X_train)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y_train)
        prediction = pol_reg.predict(poly_reg.fit_transform(X_test))
        for i in range(test_row_count):
            if prediction[i] > 5:
                prediction[i] = 5
            elif prediction[i] < 1:
                prediction[i] = 1
        avg_mean = 0
        avg_mean_sq = 0
        for i in range(test_row_count):
            mean = abs(prediction[i] - y_test[i])
            mean_sq = (abs(prediction[i] - y_test[i])) ** 2
            print("Predicted rating:", prediction[i], "Actual rating:", y_test[i], "Mean:", mean, "Mean^2:", mean_sq)
            txt = "Predicted rating: " + str(prediction[i]) + " Actual rating: " + str(y_test[i]) + " Mean: " + str(
                mean) + " Mean^2: " + str(mean_sq)
            f.write(txt)
            f.write("\n")
            avg_mean += mean
            avg_mean_sq += mean_sq
        avg_mean = avg_mean / test_row_count
        avg_mean_sq = avg_mean_sq / test_row_count
        print("Average mean:", avg_mean, "Average mean^2:", avg_mean_sq)
        txt = "Average mean: " + str(avg_mean) + " Average mean^2: " + str(avg_mean_sq)
        f.write(txt)


def linear_regression_predict_input():
    train = pd.read_csv("train_sentiment.csv")
    predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
    X_train = train[predictors]
    y_train = train["Rating"]
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    while (1):
        review = input("Type your review: ")
        X_test = sentiment_analysis(review)
        prediction = lm.predict(X_test)
        if prediction > 5:
            prediction = 5
        elif prediction < 1:
            prediction = 1
        print(prediction[0])


def logistic_regression_predict_input():
    train = pd.read_csv("train_sentiment.csv")
    predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
    X_train = train[predictors]
    y_train = train["Rating"]
    lm = LogisticRegression(solver='newton-cg')
    lm.fit(X_train, y_train)
    while (1):
        review = input("Type your review: ")
        X_test = sentiment_analysis(review)
        prediction = lm.predict(X_test)
        print(prediction[0])

def polynomial_regression_predict_input():
    train = pd.read_csv("train_sentiment.csv")
    predictors = ["Polarity", "Subjectivity", "Positive", "Neutral", "Negative"]
    X_train = train[predictors]
    y_train = train["Rating"]
    poly_reg = PolynomialFeatures(degree=5)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    while (1):
        review = input("Type your review: ")
        X_test = sentiment_analysis(review)
        prediction = pol_reg.predict(poly_reg.fit_transform(X_test))
        if prediction > 5:
            prediction = 5
        elif prediction < 1:
            prediction = 1
        print(prediction[0])


# test()
# train()
# linear_regression_predict()
# logistic_regression_predict()
# polynomial_regression_predict()
linear_regression_predict_input()
# logistic_regression_predict_input()
# polynomial_regression_predict_input()
