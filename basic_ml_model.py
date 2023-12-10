import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

def get_data():
    # URL = "https://raw.githubusercontent.com/amberkakkar01/Prediction-of-Wine-Quality/master/winequality-red.csv"
    
    # read data as dataframe
    try:
        df = pd.read_csv(r"C:\Users\ravikant.purnchand\Downloads\mlflow\WineQT.csv")
        df.drop(['Id'], axis = 1, inplace = True)
        return df
    except Exception as e:
        raise e

def evaluate(y_true, y_pred):
    # mae = mean_absolute_error(y_true,y_pred)
    # mse = mean_squared_error(y_true,y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    # r2 = r2_score(y_true,y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy
    # return mae, mse, rmse, r2

def main():
    df = get_data()
    train, test = train_test_split(df)

    X_train = train.drop(['quality'], axis = 1)
    X_test = test.drop(['quality'], axis = 1)

    y_train = train[['quality']]
    y_test = test[['quality']]

    # lr = ElasticNet()
    lr = RandomForestClassifier()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    #Evaluate the model
    # mae, mse, rmse, r2 = evaluate(y_test, pred)
    accuracy= evaluate(y_test, pred)

    # print(f"mean absolute error {mae}, mean squared error {mse}, root mean square error {rmse}, r2_square {r2}")
    print(f"accuracy {accuracy}")

    
if __name__ == '__main__':
    try:
        main()
        print("It's finished successfully")
    except Exception as e:
        raise e
    