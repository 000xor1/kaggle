# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import setting as st
import sigmoid_model as sm


def vote(data):
    result = []

    for i in xrange(len(data)):
        sum_value = 0
        rod = 0

        for j in xrange(len(data[0])):
            sum_value += int(data[i][j])

        rod = float(sum_value / len(data[0]))

        result.append(np.ceil(rod))

    return result


def test():

    data_x = pd.read_csv(st.TEST_X)

    passenger_id = data_x['PassengerId']
    data_x = data_x.drop(['PassengerId'], axis=1)

    data_x = np.asarray(data_x.values, np.float32)

    test_data_x = data_x

    prediction_list = np.empty((len(test_data_x), 0), float)

    # using the INPUT MODEL
    for count in xrange(st.IN_MODEL_NUM):

        model = sm.RestoreModel(st.IN_DIR+"models/model_"+str(count+1), rod=False)
        result = model.run(test_data_x)

        p = np.reshape(result, (len(test_data_x), -1))

        prediction_list = np.append(prediction_list, result[0], axis=1)

        # graph initialize
        tf.reset_default_graph()

    test_data_x = prediction_list

    prediction_list = np.empty((len(test_data_x), 0), float)

    # using the OUTPUT MODEL
    for count in xrange(st.OUT_MODEL_NUM):

        model = sm.RestoreModel(st.OUT_DIR+"models/model_"+str(count+1), rod=True)
        result = model.run(test_data_x)

        p = np.reshape(result, (-1, len(test_data_x)))
        prediction_list = np.append(prediction_list, result[0], axis=1)

        # graph initialize
        tf.reset_default_graph()

    result = vote(prediction_list)

    submission = pd.DataFrame({"PassengerId": passenger_id, "Survived": result})
    submission.Survived = submission.Survived.astype(int)
    submission.to_csv(st.SUBMMIT+"submission.csv", index=False)


if __name__ == '__main__':
    test()
