import time
import numpy as np
import pandas as pd
from scipy.stats import mode

def euclidean(p1, p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def predict(x_train, y_train, x_input, k):
    result = []

    for i in x_input:
        #point_dist = []
        #distances = [euclidean(x_train[j, :], i) for j in range(len(x_train))]
        #distances = np.array(distances)
        distances = np.fromiter((euclidean(x_train[j, :], i) for j in range(len(x_train))), dtype=float)
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
    with open('knn2_NEW.txt', 'a') as f:
        f.write(str(t2-t1) + "\n")
