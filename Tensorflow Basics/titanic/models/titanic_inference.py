import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
import io
import requests
import math
from scipy import stats


from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn
import json
import math


def sigmoid(x) :

    lis = []
    for i in x :
        lis.append(1 / (1 + math.exp(-i)))

    return lis


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)

    print(df.head(3))
    return (dataset - mu) / sigma


def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)

    print(df.head())
    return df


def bins_details(df) :

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pd_bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9',
               '0.9-1.0']
    lis = []
    percentages = []
    cumul = []
    round_digits = 3

    for lab in labels:
        lis.append(np.sum(df["label_bins"] == lab))
    for count in lis:
        per = (count / sum(lis))
        #per=round(float(per), round_digits)
        per*= 100
        per=round(per, round_digits)
        percentages.append(per)

    for i, _ in enumerate(percentages):
        s = 0
        if (i == 0):
            cumul.append(round(percentages[i],round_digits))
        else:
            for j in range(0, i + 1):
                s += round(percentages[j],round_digits)

            s = round(s, round_digits)
            if(s<100.000) :
                cumul.append(s)
            else :
                cumul.append(100.000)

    dic = {"pd_bins": pd_bins, "counts": lis, "percentage": percentages, "cumulative":cumul}

    return dic




def pre_processing(df):
    df.head(1)
    df.drop(delete_columns, axis=1, inplace=True)
    # Count room nubmer
    # df_train["Cabin"] = df_train["Cabin"].apply(count_space_except_nan)
    # Replace NaN with mean value
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    # Pclass, Embarked one-hot
    df = one_hot(df, df.loc[:, ["Pclass"]].columns)
    # String to int
    df = str_to_int(df)

    return df


test_result_csv_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/test/result_csvs/'
test_result_bins_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/test/result_bins/'

cols1 = ['Sex','Age']
cols2 = ['SibSp','Parch']
cols3 = ['Fare','Pclass_1','Pclass_2','Pclass_3']

df_test = pd.read_csv('/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/test.csv')


delete_columns = ["Ticket", "Name", "PassengerId", "Cabin", "Embarked"]

test_passenger_id = df_test["PassengerId"].values
df_test = pre_processing(df_test)

test_1 = df_test[cols1].values
test_2 = df_test[cols2].values
test_3 = df_test[cols3].values

saved_model = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/saved_model/'
ckpt = tf.train.latest_checkpoint(saved_model)
filename = ".".join([ckpt, 'meta'])
model_saver = tf.train.import_meta_graph(filename, clear_devices=True)

# Bins
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

with tf.Session() as sess:
    model_saver.restore(sess, ckpt)
    graph = tf.get_default_graph()
    train_1_X = graph.get_tensor_by_name("placeholders/train_1:0")
    train_2_X = graph.get_tensor_by_name("placeholders/train_2:0")
    train_3_X = graph.get_tensor_by_name("placeholders/train_3:0")

    y_ = tf.get_collection('y_')[0]
    sigmoided_pred = tf.get_collection('sigmoided_y_')[0]

    logits, predictions = sess.run([y_, sigmoided_pred],
                                   feed_dict={train_1_X: test_1,
                                              train_2_X: test_2,
                                              train_3_X: test_3})


    # OOT
    df = pd.DataFrame({"PREDICTIONS": predictions.flatten(),
                       "PRIMARY_ID": test_passenger_id.flatten()})
    df['decile'] = pd.qcut(df['PREDICTIONS'], 10, labels=False)
    df.to_csv(test_result_csv_path + 'test.csv', index=False)

    df1 = pd.DataFrame()
    df1['label_bins'] = pd.cut(df['PREDICTIONS'], bins=bins, labels=labels)
    bin_data = bins_details(df1)
    df_bins = pd.DataFrame(bin_data)
    df_bins.to_csv(test_result_bins_path + 'test_bins.csv', index=False)

