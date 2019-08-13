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


df_train = pd.read_csv('/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/train.csv')

df_test = pd.read_csv('/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/test.csv')


delete_columns = ["Ticket", "Name", "PassengerId", "Cabin", "Embarked"]

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

train_result_csv_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/train/result_csvs/'
val_result_csv_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/val/result_csvs/'
train_result_bins_csv_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/train/result_bins/'
val_result_bins_csv_path = '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/val/result_bins/'


cols1 = ['Sex','Age']
cols2 = ['SibSp','Parch']
cols3 = ['Fare','Pclass_1','Pclass_2','Pclass_3']

label_data = df_train['Survived']


combined_data_train_X, combined_data_val_X, combined_data_train_y, combined_data_val_y = \
    train_test_split(df_train, label_data,
                     stratify=label_data.values, test_size=0.2, random_state=10)


#save PassengerId for evaluation
train_passenger_ids = combined_data_train_X["PassengerId"].values
val_passenger_ids = combined_data_val_X["PassengerId"].values



df_train = pre_processing(combined_data_train_X)
df_val = pre_processing(combined_data_val_X)


# df_val = combined_data_val_X
val_label = combined_data_val_y
val_label = np.reshape(val_label.values, newshape=(len(val_label), 1))

df_train_1, df_val_1 = df_train[cols1].values, df_val[cols1].values
df_train_2, df_val_2 = df_train[cols2].values, df_val[cols2].values
df_train_3, df_val_3 = df_train[cols3].values, df_val[cols3].values



def divide_batches(input_batch, batch_size):
    """
    Divide into batches

    :param input_batch:
    :param batch_size:
    :return:
    """
    output_batch = []
    for i in range(0, len(input_batch), batch_size):
        output_batch.append(input_batch[i: i + batch_size])
    return output_batch



batch_size = 33

features = df_train.iloc[:, 1:].values
# features = feature_normalize(features)
train_label = df_train.iloc[:, :1]
train_label = np.reshape(train_label.values, newshape=(len(train_label), 1))

test_label = df_test.iloc[:, :1]
test_label = np.reshape(test_label.values, newshape=(len(test_label), 1))

df_train_1_x = divide_batches(df_train_1, batch_size)
df_val_1_x = divide_batches(df_val_1, batch_size)

df_train_2_x = divide_batches(df_train_2, batch_size)
df_val_2_x = divide_batches(df_val_2, batch_size)

df_train_3_x = divide_batches(df_train_3, batch_size)
df_val_3_x = divide_batches(df_val_3, batch_size)

train_y = divide_batches(train_label, batch_size)
val_y = divide_batches(val_label, batch_size)


keep_probability = 0.5



# Placeholders.
with tf.name_scope("placeholders"):
    train_1_X = tf.placeholder(dtype=tf.float32, shape=[None, len(df_train_1[0])], name="train_1")
    train_2_X = tf.placeholder(dtype=tf.float32, shape=[None, len(df_train_2[0])], name="train_2")
    train_3_X = tf.placeholder(dtype=tf.float32, shape=[None, len(df_train_3[0])], name="train_3")
    y = tf.placeholder(dtype=tf.float32, shape=[None, len(train_label[0])], name="output")
    lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    z = tf.placeholder(dtype=tf.float32, shape=[], name="z")


def model(train_1_x_inp, train_2_x_inp, train_3_x_inp):
    with tf.variable_scope('model_weights'):
        model_weights = {
            'dynamic_ffn_w1': tf.get_variable(name="dynamic_ffn_w1", shape=[30, 50],
                                              initializer=tf.initializers.random_normal),
            'train_3_h1_wt': tf.get_variable(name="train_3_h1_wt", shape=[len(df_train_3[0]), 10],
                                             initializer=tf.initializers.random_normal),

            'ffn_h1_wt': tf.get_variable(name="ffn_h1_wt", shape=[50, 50],
                                         initializer=tf.initializers.random_normal),
            'ffn_h2_wt': tf.get_variable(name="ffn_h2_wt", shape=[50, 30],
                                         initializer=tf.initializers.random_normal),
            'ffn_out_wt': tf.get_variable(name="ffn_out_wt", shape=[30, 1],
                                          initializer=tf.initializers.random_normal)
        }

    with tf.variable_scope('model_biases'):
        model_bias = {
            'dynamic_ffn_b1': tf.get_variable(name="dynamic_ffn_b1", shape=[50],
                                              initializer=tf.initializers.random_normal),
            'train_3_h1_bias': tf.get_variable(name="train_3_h1_bias", shape=[10],
                                               initializer=tf.initializers.random_normal),
            'ffn_h1_bias': tf.get_variable(name="ffn_h1_bias", shape=[50],
                                           initializer=tf.initializers.random_normal),
            'ffn_h2_bias': tf.get_variable(name="ffn_h2_bias", shape=[30],
                                           initializer=tf.initializers.random_normal),
            'ffn_out_bias': tf.get_variable(name="ffn_out_bias", shape=[1],
                                            initializer=tf.initializers.random_normal)
        }

    with tf.variable_scope('train_12_weights_biases'):
        model_train_12_weights = {'train_12_lstm_w1': tf.get_variable(name="train_12_lstm_w1", shape=[20, 20],
                                                                      initializer=tf.initializers.random_normal)}
        model_train_12_biases = {'train_12_lstm_b1': tf.get_variable(name="train_12_lstm_b1", shape=[20],
                                                                     initializer=tf.initializers.random_normal)}

    with tf.name_scope("DL_Model"):

        with tf.name_scope('train_1_lstm'):
            # 1st lstm
            reshape_train_1_data = tf.reshape(train_1_x_inp, name="reshape_train_1",
                                              shape=[-1, len(df_train_1[0]), 1])
            unstack_train_1_data = tf.unstack(reshape_train_1_data, name="unstack_trans", axis=1)
            train_1_lstm_cell = rnn.BasicLSTMCell(name="train_1_lstm", num_units=10, activation=tf.nn.relu)
            train_1_lstm, train_1_lstm_states = rnn.static_rnn(train_1_lstm_cell, unstack_train_1_data,
                                                               dtype=tf.float32)

        with tf.name_scope('train_2_lstm'):
            # 2nd lstm
            reshape_train_2_data = tf.reshape(train_2_x_inp, name="reshape_train_2",
                                              shape=[-1, len(df_train_2[0]), 1])
            unstack_train_2_data = tf.unstack(reshape_train_2_data, name="unstack_trans", axis=1)
            train_2_lstm_cell = rnn.BasicLSTMCell(name="train_2_lstm", num_units=10, activation=tf.nn.relu)
            train_2_lstm, train_2_lstm_states = rnn.static_rnn(train_2_lstm_cell, unstack_train_2_data,
                                                               dtype=tf.float32)

        with tf.name_scope('concat_train_1_2'):
            # concat train_1 and credit train_2
            concatenated_lstm_train_12 = tf.concat((train_1_lstm[-1], train_2_lstm[-1]), axis=1)
            train_ffn1 = tf.nn.sigmoid(tf.add(
                tf.matmul(concatenated_lstm_train_12, model_train_12_weights['train_12_lstm_w1']),
                model_train_12_biases[
                    'train_12_lstm_b1']))

        with tf.name_scope('train_3_ffn'):
            train_3_h1 = tf.add(tf.matmul(train_3_x_inp, model_weights['train_3_h1_wt']),
                                model_bias['train_3_h1_bias'])
            train_3_h1 = tf.nn.sigmoid(train_3_h1, name='train_3_h1_activation')

        with tf.name_scope('concat_train_12_train_3'):
            # concat credit train_12_ffn1, train_3_ffn
            trains_nw = tf.concat((train_ffn1, train_3_h1), axis=1)  # 30 features

        with tf.name_scope('combined_ffn'):
            ffn1 = tf.add(tf.matmul(trains_nw, model_weights['dynamic_ffn_w1']), model_bias['dynamic_ffn_b1'])
            ffn1 = tf.nn.sigmoid(ffn1, name='activated_ffn1')
            ffn1 = tf.nn.dropout(ffn1, keep_prob=keep_probability)

            ffn2 = tf.add(tf.matmul(ffn1, model_weights['ffn_h1_wt']), model_bias['ffn_h1_bias'])
            ffn2 = tf.nn.sigmoid(ffn2, name='activated_ffn1')
            ffn2 = tf.nn.dropout(ffn2, keep_prob=keep_probability)

            ffn3 = tf.add(tf.matmul(ffn2, model_weights['ffn_h2_wt']), model_bias['ffn_h2_bias'])
            ffn3 = tf.nn.sigmoid(ffn3, name='activated_ffn2')

            ffn_out = tf.add(tf.matmul(ffn3, model_weights['ffn_out_wt']), model_bias['ffn_out_bias'],
                             name='logits')

    return ffn_out



y_ = model(train_1_X, train_2_X, train_3_X)


sigmoided_y_ = tf.nn.sigmoid(y_, name='ffn_out_activation')
tf.add_to_collection("y_", y_)
tf.add_to_collection("sigmoided_y_", sigmoided_y_)

# Loss.
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y),
                          name='model_loss')
    tf.add_to_collection("loss", loss)

# Optimizer.
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    tf.add_to_collection("optimizer", optimizer)

train_avg_loss_summ = tf.summary.scalar("train_avg_loss", z)
val_avg_loss_summ = tf.summary.scalar("val_avg_loss", z)

epochs = 35

# Bins
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

init = tf.global_variables_initializer()

# Model saver
model_saver = tf.train.Saver()

with tf.Session() as sess:

    print('Session initalised...')
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    writer = tf.summary.FileWriter('/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/tensorboard_summary/', sess.graph)
    writer.add_graph(sess.graph)

    train_count = 0
    val_count = 0
    print('Training started..')

    for i in range(1, epochs + 1):

        print("Epoch", i)
        train_set = zip(df_train_1_x,df_train_2_x,df_train_3_x,train_y)
        val_set = zip(df_val_1_x,df_val_2_x,df_val_3_x,val_y)

        # Train Data.
        count = 0
        train_loss = []
        train_predictions = []

        for train_1_batch,train_2_batch,train_3_batch,train_label_batch in train_set:
            train_count += 1
            count += 1

            _, l = sess.run([optimizer, loss], feed_dict={train_1_X: train_1_batch,
                                                         train_2_X: train_2_batch,
                                                         train_3_X: train_3_batch,
                                                         y: train_label_batch,
                                                         lr: 0.001})

            model_prediction = sess.run(y_, feed_dict={train_1_X: train_1_batch,
                                                         train_2_X: train_2_batch,
                                                         train_3_X: train_3_batch,
                                                         y: train_label_batch,
                                                         lr: 0.001})

            train_loss.append(l)
            train_predictions.append(temp for temp in model_prediction)



        # Train Data.
        count = 0
        val_loss = []
        val_predictions = []

        for val_1_batch,val_2_batch,val_3_batch,val_label_batch in val_set:
            train_count += 1
            count += 1

            l = sess.run([loss], feed_dict={train_1_X: val_1_batch,
                                                         train_2_X: val_2_batch,
                                                         train_3_X: val_3_batch,
                                                         y: val_label_batch,
                                                         lr: 0.001})

            model_prediction = sess.run(y_, feed_dict={train_1_X: val_1_batch,
                                                         train_2_X: val_2_batch,
                                                         train_3_X: val_3_batch,
                                                         y: val_label_batch,
                                                         lr: 0.001})

            val_loss.append(l)
            val_predictions.append(temp for temp in model_prediction)



        train_predictions = [item for sublist in train_predictions for item in sublist]
        val_predictions = [item for sublist in val_predictions for item in sublist]

        trans_train_label = np.asarray([item for sublist in train_y for item in sublist]).flatten()
        trans_val_label = np.asarray([item for sublist in val_y for item in sublist]).flatten()


        train_predictions = np.asarray(train_predictions).flatten()
        val_predictions = np.asarray(val_predictions).flatten()



        z_temp = sess.run(train_avg_loss_summ, feed_dict={z: sum(train_loss) / len(train_loss)})
        writer.add_summary(z_temp, i)

        z_temp = sess.run(val_avg_loss_summ, feed_dict={z: sum(train_loss) / len(train_loss)})
        writer.add_summary(z_temp, i)

        model_saver.save(sess, '/Users/ayanbask/PycharmProjects/First/Tensorflow Basics/titanic/summary_results/saved_model/', global_step=i)



        # Train
        sigmoided_train_pred = sigmoid(train_predictions)
        df = pd.DataFrame(
            {"PREDICTIONS": train_predictions, "LABEL": trans_train_label,
             "SIGMOIDED_PRED": sigmoided_train_pred, "PRIMARY_ID": train_passenger_ids.flatten()})
        df['decile'] = pd.qcut(df['SIGMOIDED_PRED'], 10, labels=False)
        df['label_bins'] = pd.cut(df['SIGMOIDED_PRED'], bins=bins, labels=labels)
        df.to_csv(train_result_csv_path + str(i) + '.csv', index=False)

        bin_train_data = bins_details(df)
        df_bins_train = pd.DataFrame(bin_train_data)
        df_bins_train.to_csv(train_result_bins_csv_path + str(i) + '.csv', index=False)

        # Test
        sigmoided_val_pred = sigmoid(val_predictions)
        df = pd.DataFrame({"PREDICTIONS": val_predictions, "LABEL": trans_val_label,
                           "SIGMOIDED_PRED": sigmoided_val_pred,
                           "PRIMARY_ID": val_passenger_ids.flatten()})
        df['decile'] = pd.qcut(df['SIGMOIDED_PRED'], 10, labels=False)
        df['label_bins'] = pd.cut(df['SIGMOIDED_PRED'], bins=bins, labels=labels)
        df.to_csv(val_result_csv_path + str(i) + '.csv', index=False)

        bin_val_data = bins_details(df)
        df_bins_val = pd.DataFrame(bin_val_data)
        df_bins_val.to_csv(val_result_bins_csv_path + str(i) + '.csv', index=False)