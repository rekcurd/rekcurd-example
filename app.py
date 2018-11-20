#!/usr/bin/python
# -*- coding: utf-8 -*-


import traceback
import csv
import os
import io

from enum import Enum
from typing import Tuple, List

from drucker.logger import JsonSystemLogger
from drucker import Drucker
from drucker.utils import PredictLabel, PredictResult, EvaluateResult, EvaluateDetail

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.externals import joblib


class MyApp(Drucker):
    def __init__(self, config_file: str = None):
        super().__init__(config_file)
        self.logger = JsonSystemLogger(self.config)
        self.load_model()

    def set_type(self, type_input: Enum, type_output: Enum) -> None:
        super().set_type(type_input, type_output)

    def get_type_input(self) -> Enum:
        return super().get_type_input()

    def get_type_output(self) -> Enum:
        return super().get_type_output()

    def load_model(self) -> None:
        """ override
        Load ML model.
        :return:
        """
        assert self.model_path is not None, \
            'Please specify your ML model path'
        try:
            self.predictor = joblib.load(self.model_path)
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            self.predictor = None
            if not self.is_first_boot():
                # noinspection PyProtectedMember
                os._exit(-1)

    def predict(self, input: PredictLabel, option: dict = None) -> PredictResult:
        """ override
        Predict.

        :param input: Input data. string/bytes/arr[int]/arr[float]/arr[string]
        :param option: Miscellaneous. dict
        :return:
            output: Result. string/bytes/arr[int]/arr[float]/arr[string]
            score: Score. float/arr[float]
            option: Miscellaneous. dict
        """
        try:
            if option is None:
                option = {}
            label_predict = self.predictor.predict(
                np.array([input], dtype='float64')).tolist()
            return PredictResult(label_predict, [1] * len(label_predict), option={})
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            raise e

    def evaluate(self, file: bytes) -> Tuple[EvaluateResult, List[EvaluateDetail]]:
        """ override
        Evaluate

        :param file: Evaluation data file. bytes
        :return:
            num: Number of data. int
            accuracy: Accuracy. float
            precision: Precision. arr[float]
            recall: Recall. arr[float]
            fvalue: F1 value. arr[float]
            option: optional metrics. dict[str, float]

            details: detail result of each prediction
        """
        try:
            f = io.StringIO(file.decode("utf-8"))
            reader = csv.reader(f, delimiter=",")
            num = 0
            label_gold = []
            label_predict = []
            details = []
            for row in reader:
                num += 1
                correct_label = int(row[0])
                label_gold.append(correct_label)
                result = self.predict(row[1:], option={})
                is_correct = correct_label == int(result.label[0])
                details.append(EvaluateDetail(input, correct_label, result, is_correct))
                label_predict.append(result.label)

            accuracy = accuracy_score(label_gold, label_predict)
            p_r_f = precision_recall_fscore_support(label_gold, label_predict)
            res = EvaluateResult(num, accuracy, p_r_f[0].tolist(), p_r_f[1].tolist(), p_r_f[2].tolist(), {})
            return res, details
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            return EvaluateResult(), []
