import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(data):

    print(f'Tres registros al azar: \n{data.sample(3)}\n\n')
    print(f'Descripcion estadistica:\n{data.describe()}\n\n')
    print(f'Info relevante: \n{data.info()}\n\n')

    dailySale = data.groupby('date')['total_amount'].sum()


    plt.figure(figsize=(14, 7))
    plt.plot(dailySale, marker='o')
    plt.title('Ventas Diarias')
    plt.xlabel('Fecha')
    plt.ylabel('Monto total')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


    numericalData = data.select_dtypes(include=['float64', 'int64']).columns
    data[numericalData].hist(figsize=(7, 5), bins=20)
    plt.tight_layout()
    plt.show()

    sns.heatmap(data[numericalData].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()


def loadData(path):

    data = pd.read_csv(path)
    return data

def prepData(data):

    data['date'] = pd.to_datetime(data['date'])
    data['dayYear'] = data['date'].dt.dayofyear
    return data

def trainModel(data):

    X = data[['dayYear']]
    y = data['total_amount']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(Xtrain, ytrain)
    return model

def generatePredict(model, data, days=30):

    futureDates = pd.date_range(start=data['date'].max() + timedelta(days=1), periods=days)
    futureDays = futureDates.dayofyear
    futureX = pd.DataFrame(futureDays, columns=['dayYear'])
    predictions = model.predict(futureX)
    predictDF = pd.DataFrame({'date': futureDates, 'predict_Total_amount': predictions})
    return predictDF

def savePredict(predictDF, path):

    predictDF.to_csv(path, index=False)
    return path

def main():
    path = 'dataset.csv'
    data = loadData(path)
    data = prepData(data)
    model = trainModel(data)

    dataAnalysis(data)

    days = 30
    predictDF = generatePredict(model, data, days)
    predictions_file_path = 'predictions.csv'
    savePredict(predictDF, predictions_file_path)
    print(f' Predicciones para los proximos 10 dias: \n{predictDF.head(10)}\n\n')
    print(predictions_file_path)

if __name__ == "__main__":
    main()
