#!/usr/bin/python
# -*- coding: utf-8 -*-

import traceback
import io

from enum import Enum

from drucker.logger.logger_jsonlogger import SystemLogger
from drucker.core.predict_interface import PredictInterface, PredictLabel, PredictResult, EvaluateResult
from drucker.utils.env_loader import SERVICE_LEVEL_ENUM, APPLICATION_NAME
from drucker.models import get_model_path, SERVICE_FIRST_BOOT

import numpy as np
import csv
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.externals import joblib


class Predict(PredictInterface):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.logger = SystemLogger(logger_name="drucker_predict", app_name=APPLICATION_NAME, app_env=SERVICE_LEVEL_ENUM)
        self.load_model(get_model_path())

    def set_type(self, type_input: Enum, type_output: Enum) -> None:
        """@deprecated
        """
        super().set_type(type_input, type_output)

    def get_type_input(self) -> Enum:
        """@deprecated
        """
        return super().get_type_input()

    def get_type_output(self) -> Enum:
        """@deprecated
        """
        return super().get_type_output()

    def load_model(self, model_path: str = None) -> None:
        """ override
        Load ML model.

        :param model_path:
        :return:
        """
        assert model_path is not None, \
            'Please specify your ML model path'
        try:
            self.predictor = joblib.load(model_path)

        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            self.predictor = None
            if not SERVICE_FIRST_BOOT:
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

    def evaluate(self, file: bytes) -> EvaluateResult:
        """ override
        [WIP] Evaluate.

        :param file: Evaluation data file. bytes
        :return:
            num: Number of data. int
            accuracy: Accuracy. float
            precision: Precision. arr[float]
            recall: Recall. arr[float]
            fvalue: F1 value. arr[float]
        """
        try:
            f = io.StringIO(file.decode("utf-8"))
            reader = csv.reader(f, delimiter=",")
            num = 0
            label_gold = []
            label_predict = []
            for row in reader:
                num += 1
                label_gold.append(int(row[0]))
                result = self.predict(row[1:], option={})
                label_predict.append(result.label)

            accuracy = accuracy_score(label_gold, label_predict)
            p_r_f = precision_recall_fscore_support(label_gold, label_predict)
            return EvaluateResult(num, accuracy, p_r_f[0].tolist(), p_r_f[1].tolist(), p_r_f[2].tolist())
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            return EvaluateResult()
