import time
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.stats import mode
import warnings
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
warnings.filterwarnings('ignore')

def euclidean(p1, p2):
    dist = tf.sqrt(tf.reduce_sum(tf.subtract(p1,p2)**2))
    return dist

def predict(X_train, y_train, X_input, k_tf):
    result = []
    point_dist = []
    for x in X_input:
        for j in range(len(X_train)):
            distance = euclidean(X_train[j,:], x)
            point_dist.append(distance)
        distances = np.array(point_dist)
        vals, indx = tf.nn.top_k(tf.negative(distances), k_tf)
        y_s = tf.gather(y_train, indx)
        result_label = mode(y_s.numpy())
        result.append(result_label.mode[0])
    return result


training_df = pd.read_csv('trainingsample.csv')
validation_df = pd.read_csv('validationsample.csv')
X_train = training_df.drop(columns=['Class']).to_numpy()
y_train = training_df['Class'].to_numpy()
X_test = [validation_df.drop(columns=['Class']).to_numpy()[0]]
k =tf.constant(5)

for i in range(200):
    t1 = time.time()
    result = predict(X_train, y_train, X_test, k)
    t2 = time.time()
    with open('knn_tensorflow2_NEW.txt', 'a') as f:
        f.write(str(t2-t1) + "\n")