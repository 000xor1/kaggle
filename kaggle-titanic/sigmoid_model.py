# -*- coding: utf-8 -*-

import os
import sys
import functools
import tensorflow as tf
import setting as st


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# Train the model
class Model:
    def __init__(self, x, y, mode, keep_prob):
        self.X = x
        self.Y = y
        self.keep_prob = keep_prob
        self.mode = mode

        self.prediction
        self.optimize
        self.accuracy
        self.error
        self.loss

        tf.add_to_collection("X", self.X)
        tf.add_to_collection("mode", self.mode)
        tf.add_to_collection("keep_prob", self.keep_prob)

    @lazy_property
    def prediction(self):

        X = self.X

        with tf.variable_scope('layer1'):
            W1 = tf.get_variable("W1", shape=[9, 5],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("B1", shape=[5],
                                 initializer=tf.zeros_initializer())
            L1 = tf.layers.batch_normalization(tf.matmul(X, W1) + b1,
                                               training=self.mode)
            L1 = tf.nn.elu(L1)

            tf.summary.histogram("X", X)
            tf.summary.histogram("weights", W1)
            tf.summary.histogram("bias", b1)
            tf.summary.histogram("layer", L1)

        with tf.variable_scope('layer2'):
            W2 = tf.get_variable("W2", shape=[5, 5],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("B2", shape=[5],
                                 initializer=tf.zeros_initializer())
            L1 = tf.layers.batch_normalization(L1, training=self.mode)
            L2 = tf.nn.elu(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            tf.summary.histogram("weights", W2)
            tf.summary.histogram("bias", b2)
            tf.summary.histogram("layer", L2)

        with tf.variable_scope('layer3'):
            W3 = tf.get_variable("W3", shape=[5, 5],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("B3", shape=[5],
                                 initializer=tf.zeros_initializer())
            L2 = tf.layers.batch_normalization(L2, training=self.mode)
            L3 = tf.nn.elu(tf.matmul(L2, W3) + b3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            tf.summary.histogram("weights", W3)
            tf.summary.histogram("bias", b3)
            tf.summary.histogram("layer", L3)

        with tf.variable_scope('layer4'):
            W4 = tf.get_variable("W4", shape=[5, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("B4", shape=[1],
                                 initializer=tf.zeros_initializer())
            L3 = tf.layers.batch_normalization(L3, training=self.mode)

            logit = tf.matmul(L3, W4) + b4
            hypothesis = tf.nn.sigmoid(logit, name="hypothesis")

            tf.add_to_collection("hypothesis", hypothesis)

            tf.summary.histogram("weights", W4)
            tf.summary.histogram("bias", b4)
            tf.summary.histogram("hypothesis", hypothesis)

        return logit, hypothesis

    @lazy_property
    def loss(self):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.prediction[0],
            labels=self.Y), name="cost")

        tf.add_to_collection("loss", cost)

        return cost

    @lazy_property
    def optimize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer().minimize(self.loss,
                                                     name="optimizer")

    @lazy_property
    def accuracy(self):
        predicted = tf.cast(self.prediction[1] > 0.5, dtype=tf.float32)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, self.Y),
                                          dtype=tf.float32))

        return accuracy

    @lazy_property
    def error(self):
        predicted = tf.cast(self.prediction[1] > 0.5, dtype=tf.float32)

        error = tf.reduce_mean(tf.cast(tf.not_equal(predicted, self.Y),
                                       dtype=tf.float32))

        return error


class RestoreModel:

    def __init__(self, loc, rod=True):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.round = rod

        with self.graph.as_default():

            saver = tf.train.import_meta_graph(loc + "/sigmoid_model-"+str(st.EPOCH_SIZE)+".meta",
                                               clear_devices=True)

            saver.restore(self.sess, loc + "/sigmoid_model-"+str(st.EPOCH_SIZE))

            self.X = tf.get_collection("X")[0]
            self.hypothesis = tf.get_collection("hypothesis")[0]
            self.keep_prob = tf.get_collection("keep_prob")[0]
            self.mode = tf.get_collection("mode")[0]

            if self.round is True:
                self.prediction = tf.cast(self.hypothesis > 0.5,
                                          dtype=tf.float32)
            else:
                self.prediction = tf.cast(self.hypothesis,
                                          dtype=tf.float32)

    def run(self, data):
        """ Running the activation function previously imported """
        return self.sess.run([self.prediction],
                             feed_dict={self.X: data, self.mode: False,
                                        self.keep_prob: 1.0})
