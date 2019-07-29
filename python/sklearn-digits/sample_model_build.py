#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from app import MyApp


app = MyApp()
app.load_config_file("./settings.yml")
output_filepath = Path(app.config.MODEL_FILE_PATH)
local_mode = (app.config.MODEL_MODE_ENUM.value == 'local')


def run():
    digits = load_digits()
    data_train, data_test, label_train, label_test = \
        train_test_split(digits.data, digits.target, test_size=0.25, shuffle=True)
    estimator = LinearSVC(C=1.0)
    estimator.fit(data_train, label_train)
    print(estimator)

    label_predict = estimator.predict(data_test)
    print(accuracy_score(label_test, label_predict))
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, str(output_filepath))


if __name__ == '__main__':
    if local_mode and not output_filepath.exists():
        run()
    else:
        print("No need to create model. Finish!")
