import time
import numpy as np
import pandas as pd
from scipy.stats import mode


def predict(x_train, y_train, x_input, k):
    result = []

    for i in x_input:
        distances = []
        distances  = np.sqrt(np.sum((np.subtract(x_train, x_input[0])**2), axis=1))
        #distances  = np.sqrt(np.sum((np.square(np.subtract(x_train, x_input[0]))), axis=1))
        dist = np.argsort(distances)[:k]
        labels = y_train[dist]
        result_label = mode(labels)
        result_label = result_label.mode[0]
        result.append(result_label)

    return result

training_df = pd.read_csv('trainingsample.csv')
validation_df = pd.read_csv('validationsample.csv')
X_train = training_df.drop(columns=['Class']).to_numpy()
y_train = training_df['Class'].to_numpy()
X_test = [validation_df.drop(columns=['Class']).to_numpy()[0]]


for i in range(200):
    t1 = time.time()
    result = predict(X_train, y_train, X_test, 5)
    t2 = time.time()
    with open('knn_NEW.txt', 'a') as f:
        f.write(str(t2-t1) + "\n")