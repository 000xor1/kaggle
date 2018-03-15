# -*- coding: utf-8 -*-
# setting
import os

PATH = os.path.dirname(os.path.realpath(__file__))


IN_MODEL_NUM = 9
OUT_MODEL_NUM = 10

# default: 20
BATCH_SIZE = 40

# default: 200
EPOCH_SIZE = 200

# Not preprocess the data set.
TRAIN = PATH+'/data/train.csv'
TEST = PATH+'/data/test.csv'
Gender = PATH+'./data/gender_submission.csv'

TB_SUMMARY_DIR = PATH+'/save/logs/'
TF_MODEL_DIR = PATH+'/save/models/'

# result values from preprocessing.py
TRAIN_X = PATH+'/data/p_train_x.csv'
TRAIN_Y = PATH+'/data/p_train_y.csv'
TEST_X = PATH+'/data/p_test_x.csv'

IN_DIR = PATH+'/save/in/'
OUT_DIR = PATH+'/save/out/'

SUBMMIT = PATH+'/result/'
