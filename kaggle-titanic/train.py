# -*- coding: utf-8 -*-

import os
import sys
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import setting as st
import sigmoid_model as sm


def next_batch(batch, data, batch_size):
    if batch is 0:
        data = data[0:batch_size]

    elif batch is not 0:
        if len(data)-(batch_size*batch) < batch_size:
            data = data[batch_size*batch:]

        else:
            data = data[batch_size*batch:batch_size*(batch+1)]

    return data


def random_shuffle(data, num):
    random.seed(num)
    data = random.sample(data, len(data))

    return data


def merge(data_x):
    data_num = len(data_x)
    prediction_list = np.empty((data_num, 0), float)

    print "\nUsing graph models to merge the data set...\n"

    for count in xrange(st.IN_MODEL_NUM):

        model = sm.RestoreModel(st.IN_DIR+"/models/model_"+str(count+1),
                                rod=False)
        result = model.run(data_x)
        p = np.reshape(result, (data_num, -1))
        prediction_list = np.append(prediction_list, result[0], axis=1)

        tf.reset_default_graph()

    return prediction_list


def train(data_x, data_y, m, d, count, directory):

    X = tf.placeholder(tf.float32, shape=[None, 9], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

    mode = tf.placeholder(tf.bool, name='mode')

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    model = sm.Model(X, Y, mode, keep_prob)

    tf.summary.scalar("accuracy", model.accuracy)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("error", model.error)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=None)

    model_directory = directory+"models/model_"+str(count+1)+"/"
    log_directory = directory+"logs/log_"+str(count+1)+"/"

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    global_step = 0

    seed_num = random.randint(1, 100000)

    train_data_x = random_shuffle(data_x, seed_num)
    train_data_y = random_shuffle(data_y, seed_num)

    # train_data_x = train_data_x[:450]
    # train_data_y = train_data_y[:450]

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(log_directory+'/train_writer',
                                             sess.graph)

        for epoch in range(st.EPOCH_SIZE):

            train_avg_cost = 0
            train_avg_accuracy = 0

            if int(len(train_data_x) % st.BATCH_SIZE) == 0:
                total_batch = int(len(train_data_x) / st.BATCH_SIZE)
            else:
                total_batch = int(len(train_data_x) / st.BATCH_SIZE) + 1

            for batch in range(total_batch):

                batch_x = next_batch(batch, train_data_x, st.BATCH_SIZE)
                batch_y = next_batch(batch, train_data_y, st.BATCH_SIZE)

                s, a, l, _ = sess.run([
                    summary, model.accuracy, model.loss, model.optimize],
                    feed_dict={X: batch_x, Y: batch_y,
                               mode: m, keep_prob: d})

                train_avg_cost += l / total_batch
                train_avg_accuracy += a / total_batch

            global_step += 1

            train_writer.add_summary(s, global_step=global_step)
            train_writer.flush()

            if ((epoch+1) % 20) == 0:
                save_path = saver.save(sess, model_directory+"sigmoid_model",
                                       epoch+1)
                saver.export_meta_graph(model_directory+"sigmoid_model")

                print ("Epoch : {}/{}".format(epoch+1, st.EPOCH_SIZE),
                       " accuracy: "'{:f}'.format(train_avg_accuracy),
                       " loss: "'{:f}'.format(train_avg_cost),
                       " save path: ", [save_path],
                       " seed num: ", [seed_num])


if __name__ == '__main__':

    data_x = pd.read_csv(st.TRAIN_X)
    data_y = pd.read_csv(st.TRAIN_Y)

    data_x = np.asarray(data_x.values, np.float32)
    data_y = np.asarray(data_y.values, np.float32)

    for model_num in xrange(st.IN_MODEL_NUM):
        train(data_x, data_y, True, 1.0, model_num, st.IN_DIR)
        tf.reset_default_graph()

        print "\nModels : {}/{}".format(model_num+1, st.IN_MODEL_NUM)+" :: Successfully trained\n"

    prediction = merge(data_x)

    for model_num in xrange(st.OUT_MODEL_NUM):
        train(prediction, data_y, True, 1.0, model_num, st.OUT_DIR)
        tf.reset_default_graph()

        print "\nModels : {}/{}".format(model_num+1, st.OUT_MODEL_NUM)+" :: Successfully trained\n"
