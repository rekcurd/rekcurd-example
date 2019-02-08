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

        self.labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.label2idx = dict((l, i) for i, l in enumerate(self.labels))
        self.idx2label = dict((i, l) for l, i in self.label2idx.items())

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
            return PredictResult(self.idx2label[label_predict[0]], 1, option={})
        except Exception as e:
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            raise e

    def __generate_eval_data(self, file_path: str) -> Generator[Tuple[int, List[str]], None, None]:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                yield self.label2idx[row[0]], row[1:]

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
            for correct_label_idx, data in self.__generate_eval_data(file_path):
                num += 1
                label_gold.append(correct_label_idx)
                result = self.predict(data, option={})
                predict_label_idx = self.label2idx[result.label]
                is_correct = correct_label_idx == predict_label_idx
                details.append(EvaluateResultDetail(result, is_correct))
                label_predict.append(predict_label_idx)

            accuracy = accuracy_score(label_gold, label_predict)
            prf_label_order = [self.label2idx[l] for l in self.labels]
            p_r_f = precision_recall_fscore_support(label_gold, label_predict, labels=prf_label_order)
            res = EvaluateResult(num, accuracy, p_r_f[0].tolist(), p_r_f[1].tolist(), p_r_f[2].tolist(), self.labels)
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
