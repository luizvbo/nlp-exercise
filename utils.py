import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def preprocess(x):
    x = re.sub("<br\\s*/?>", " ", x)
    return x


def get_reviews():
    return (
        pd.read_csv("data/IMDB Dataset.csv")
        .assign(review=lambda df: df['review'].apply(preprocess))
    )


def get_x_y(df):
    X = df['review'].values
    y = (df['sentiment'] == 'positive').values
    return X, y


def results(df, make_model, n_data_points):
    X, y = get_x_y(df)

    for n_data_point in n_data_points:
        _, X_sample, _, y_sample = train_test_split(
            X, y, test_size=n_data_point - 1, random_state=0)

        model = make_model()
        print(f'Fits for number of data points:{n_data_point}')
        model.fit(X_sample, y_sample)

        print('Predicts and computes accuracy for the entire data set')
        a = accuracy_score(y, model.predict(X))

        yield {
            'n_data_points': n_data_point,
            'accuracy': a
        }