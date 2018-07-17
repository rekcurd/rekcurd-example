#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from drucker.utils.env_loader import APPLICATION_NAME, DIR_MODEL

config = yaml.load(open("settings.yml", 'r'))
_DIR_OUTPUT = '{0}/{1}/'.format(DIR_MODEL, APPLICATION_NAME)
_FILE_OUTPUT = _DIR_OUTPUT + 'default.model'


def run():
    digits = load_digits()
    data_train, data_test, label_train, label_test = \
        train_test_split(digits.data, digits.target, test_size=0.25, shuffle=True)
    estimator = LinearSVC(C=1.0)
    estimator.fit(data_train, label_train)
    print(estimator)

    label_predict = estimator.predict(data_test)
    print(accuracy_score(label_test, label_predict))
    if not os.path.isdir(_DIR_OUTPUT):
        os.makedirs(_DIR_OUTPUT)
    joblib.dump(estimator, _FILE_OUTPUT)


if __name__ == '__main__':
    if not os.path.isfile(_FILE_OUTPUT):
        run()
    else:
        print("Model already exist. Finish!")
