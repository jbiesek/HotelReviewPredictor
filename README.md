# HotelReviewPredictor
Hotel review predictor written in Python. It uses 18492 TripAdvisor reviews to train and 1999 reviews to test.
| Review | Rating     | 
| :-------- | :------- | 
| "nice boutique hotel stayed 5 nights, rooms nice clean place pretty, location good central singapore" | 5 | 

At the beggining it uses Spacy NLP kit to conduct an analysis on review's polarity and subjectivity, then it saves it in csv file together with the number of positive, neutral and negative words.

| Polarity | Subjectivity | Positive | Neutral | Negative | Rating |
| :-------- | :------- | :------- | :------- | :------- | :------- | 
|0.41944444444444445 | 0.7583333333333334 | 5 | 1 | 0 | 5 |

Having that, it creates Linear Regression or Logistic Regression model from sklearn.

#### Linear Regression test result:

  `Average mean: 0.7165238527962132 Average mean^2: 0.8444346302825958`

#### Logistic Regression test result:

  `Average mean: 0.6898449224612306 Average mean^2: 1.1520760380190096`
