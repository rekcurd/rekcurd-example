#!/usr/bin/python
# -*- coding: utf-8 -*-


import traceback
import csv
import os

from typing import Tuple, List, Generator

from rekcurd.logger import JsonSystemLogger
from rekcurd import Rekcurd
from rekcurd.utils import PredictInput, PredictResult, EvaluateResult, EvaluateDetail, EvaluateResultDetail

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.externals import joblib


class MyApp(Rekcurd):
    def __init__(self, config_file: str = None):
        super().__init__(config_file)
        self.logger = JsonSystemLogger(self.config)
        self.load_model()

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

    def predict(self, input: PredictInput, option: dict = None) -> PredictResult:
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

    def __generate_eval_data(self, file_path: str) -> Generator[Tuple[int, List[str]], None, None]:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                yield int(row[0]), row[1:]

    def evaluate(self, file_path: str) -> Tuple[EvaluateResult, List[EvaluateResultDetail]]:
        """ override
        Evaluate

        :param file_path: Evaluation data file path. str
        :return:
            num: Number of data. int
            accuracy: Accuracy. float
            precision: Precision. arr[float]
            recall: Recall. arr[float]
            fvalue: F1 value. arr[float]
            option: Optional metrics. dict[str, float]

            details: detail result of each prediction
        """
        try:
            num = 0
            label_gold = []
            label_predict = []
            details = []
            for correct_label, data in self.__generate_eval_data(file_path):
                num += 1
                label_gold.append(correct_label)
                result = self.predict(data, option={})
                is_correct = correct_label == int(result.label[0])
                details.append(EvaluateResultDetail(result, is_correct))
                label_predict.append(result.label)

            accuracy = accuracy_score(label_gold, label_predict)
            uniq_labels = list(set(label_gold))
            p_r_f = precision_recall_fscore_support(label_gold, label_predict, labels=uniq_labels)
            res = EvaluateResult(num, accuracy, p_r_f[0].tolist(), p_r_f[1].tolist(), p_r_f[2].tolist(), {}, uniq_labels)
            return res, details
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            return EvaluateResult(), []

    def get_evaluate_detail(self, file_path: str, results: List[EvaluateResultDetail]) -> Generator[EvaluateDetail, None, None]:
        """ override
        Create EvaluateDetail by merging evaluation data from file_path and EvaluateResultDetail

        :param file_path: Evaluation data file path. str
        :param results: Detail result of each prediction
        :return:
            detail: Evaluation data & result of each prediction
        """
        try:
            for i, (correct_label, data) in enumerate(self.__generate_eval_data(file_path)):
                yield EvaluateDetail(input=data, label=correct_label, result=results[i])
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
