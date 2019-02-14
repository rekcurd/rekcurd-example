#!/usr/bin/python
# -*- coding: utf-8 -*-


import traceback
import csv
import os

from typing import Tuple, List, Generator

from rekcurd import Rekcurd
from rekcurd.utils import PredictInput, PredictResult, EvaluateResult, EvaluateDetail, EvaluateResultDetail

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.externals import joblib


class MyApp(Rekcurd):
    def __init__(self):
        self.labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.label2idx = dict((l, i) for i, l in enumerate(self.labels))
        self.idx2label = dict((i, l) for l, i in self.label2idx.items())

    def load_model(self, filepath: str) -> object:
        """ override
        load_model
        :param filepath: ML model file path. str
        :return predictor: Your ML predictor object. object
        """
        try:
            return joblib.load(filepath)
        except Exception as e:
            self.system_logger.error(str(e))
            self.system_logger.error(traceback.format_exc())
            """Stop gRPC server."""
            os._exit(-1)

    def predict(self, predictor: object, idata: PredictInput, option: dict = None) -> PredictResult:
        """ override
        predict
        :param predictor: Your ML predictor object. object
        :param idata: Input data. PredictInput, one of string/bytes/arr[int]/arr[float]/arr[string]
        :param option: Miscellaneous. dict
        :return result: Result. PredictResult
            result.label: Label. One of string/bytes/arr[int]/arr[float]/arr[string]
            result.score: Score. One of float/arr[float]
            result.option: Miscellaneous. dict
        """
        try:
            label_predict = predictor.predict(
                np.array([input], dtype='float64')).tolist()
            return PredictResult(self.idx2label[label_predict[0]], 1, option={})
        except Exception as e:
            self.system_logger.error(str(e))
            self.system_logger.error(traceback.format_exc())
            raise e

    def __generate_eval_data(self, file_path: str) -> Generator[Tuple[int, List[str]], None, None]:
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                yield self.label2idx[row[0]], row[1:]

    def evaluate(self, predictor: object, filepath: str) -> Tuple[EvaluateResult, List[EvaluateResultDetail]]:
        """ override
        evaluate
        :param predictor: Your ML predictor object. object
        :param filepath: Evaluation data file path. str
        :return result: Result. EvaluateResult
            result.num: Number of data. int
            result.accuracy: Accuracy. float
            result.precision: Precision. arr[float]
            result.recall: Recall. arr[float]
            result.fvalue: F1 value. arr[float]
            result.option: Optional metrics. dict[str, float]
        :return detail[]: Detail result of each prediction. List[EvaluateResultDetail]
            detail[].result: Prediction result. PredictResult
            detail[].is_correct: Prediction result is correct or not. bool
        """
        try:
            num = 0
            label_gold = []
            label_predict = []
            details = []
            for correct_label_idx, data in self.__generate_eval_data(filepath):
                num += 1
                label_gold.append(correct_label_idx)
                result = self.predict(predictor, data, option={})
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
            self.system_logger.error(str(e))
            self.system_logger.error(traceback.format_exc())
            return EvaluateResult(), []

    def get_evaluate_detail(self, filepath: str, details: List[EvaluateResultDetail]) -> Generator[EvaluateDetail, None, None]:
        """ override
        get_evaluate_detail
        :param filepath: Evaluation data file path. str
        :param details: Detail result of each prediction. List[EvaluateResultDetail]
        :return rtn: Return results. Generator[EvaluateDetail, None, None]
            rtn.input: Input data. PredictInput, one of string/bytes/arr[int]/arr[float]/arr[string]
            rtn.label: Predict label. PredictLabel, one of string/bytes/arr[int]/arr[float]/arr[string]
            rtn.result: Predict detail. EvaluateResultDetail
        """
        try:
            for i, (correct_label, data) in enumerate(self.__generate_eval_data(filepath)):
                yield EvaluateDetail(input=data, label=[correct_label], result=details[i])
        except Exception as e:
            self.system_logger.error(str(e))
            self.system_logger.error(traceback.format_exc())


if __name__ == '__main__':
    app = MyApp()
    app.load_config_file("./settings.yml")
    app.run()
