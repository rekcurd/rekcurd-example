# Rekcurd-example python-sklearn-digits
This is the example of [Rekcurd](https://github.com/rekcurd/rekcurd-python).

## How to use it
### settings.yml
This is the configuration file of your ML module. Template is available [here](settings.yml).

### app.py
This is the Rekcurdized application of your ML module. Example code is available [here](app.py). You must implement the following methods.

#### load_model method
Implement your ML model loading method.

```python
def load_model(self) -> None:
    try:
        self.predictor = joblib.load(self.model_path)
    except Exception as e:
        self.logger.error(str(e))
```

If your ML module uses more than two files, you need to create a compressed file which includes the files it requires. And implement the method like the below.

```python
def joblib_load_from_zip(self, zip_name: str, file_name: str):
    with zipfile.ZipFile(zip_name, 'r') as zf:
        with zf.open(file_name, 'r') as zipmodel:
            return joblib.load(io.BufferedReader(io.BytesIO(zipmodel.read())))

def load_model(self) -> None:
    try:
        file_name = 'default.model'
        self.predictor = self.joblib_load_from_zip(self.model_path, file_name)
    except Exception as e:
        self.logger.error(str(e))
```

#### predict method
Implement your ML model predicting/inferring method.

```python
def predict(self, input: PredictLabel, option: dict = None) -> PredictResult:
    try:
        label_predict = self.predictor.predict(
            np.array([input], dtype='float64')).tolist()
        return PredictResult(label_predict, [1] * len(label_predict), option={})
    except Exception as e:
        raise e
```

##### PredictLabel
*V* is the length of feature vector.

|Field |Type |Description |
|:---|:---|:---|
|input <BR>(required) |One of below<BR>- string<BR>- bytes<BR>- string[*V*]<BR>- int[*V*]<BR>- double[*V*] |Input data for inference.<BR>- "Nice weather." for a sentiment analysis.<BR>- PNG file for an image transformation.<BR>- ["a", "b"] for a text summarization.<BR>- [1, 2] for a sales forcast.<BR>- [0.9, 0.1] for mnist data. |
|option |string| Option field. Must be json format. |

The "option" field needs to be a json format. Any style is Ok but we have some reserved fields below.

|Field |Type |Description |
|:---|:---|:---|
|suppress_log_input |bool |True: NOT print the input and output to the log message. <BR>False (default): Print the input and outpu to the log message. |
|YOUR KEY |any |YOUR VALUE |

##### PredictResult
*M* is the number of classes. If your algorithm is a binary classifier, you set *M* to 1. If your algorithm is a multi-class classifier, you set *M* to the number of classes.

|Field |Type |Description |
|:---|:---|:---|
|label<BR>(required) |One of below<BR> -string<BR> -bytes<BR> -string[*M*]<BR> -int[*M*]<BR> -double[*M*] |Result of inference.<BR> -"positive" for a sentiment analysis.<BR> -PNG file for an image transformation.<BR> -["a", "b"] for a multi-class classification.<BR> -[1, 2] for a multi-class classification.<BR> -[0.9, 0.1] for a multi-class classification. |
|score<BR>(required) |One of below<BR> -double<BR> -double[*M*] |Score of result.<BR> -0.98 for a binary classification.<BR> -[0.9, 0.1] for a multi-class classification. |
|option |string |Option field. Must be json format. |

#### evaluate method
Implement your ML model evaluating method.

```python
def evaluate(self, file_path: str) -> Tuple[EvaluateResult, List[EvaluateDetail]]:
    try:
        num = 0
        label_gold = []
        label_predict = []
        details = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                num += 1
                correct_label = int(row[0])
                label_gold.append(correct_label)
                result = self.predict(row[1:], option={})
                is_correct = correct_label == int(result.label[0])
                details.append(EvaluateDetail(result, is_correct))
                label_predict.append(result.label)

        accuracy = accuracy_score(label_gold, label_predict)
        p_r_f = precision_recall_fscore_support(label_gold, label_predict)
        res = EvaluateResult(num, accuracy, p_r_f[0].tolist(), p_r_f[1].tolist(), p_r_f[2].tolist(), {})
        return res, details
    except Exception as e:
        return EvaluateResult(), []
```

##### Input
Input is the file path of your evaluation data. The format is your favorite.

##### EvaluateResult and EvaluateDetail
`EvaluateDetail` is the details of evaluation result.

|Field |Type |Description |
|:---|:---|:---|
|result<BR>(required) |PredictResult |Prediction result. |
|is_correct<BR>(required) |bool |Correct or not. |

`EvaluateResult` is the evaluation score. *N* is the number of evaluation data. *M* is the number of classes. If your algorithm is a binary classifier, you set *M* to 1. If your algorithm is a multi-class classifier, you set *M* to the number of classes.

|Field |Type |Description |
|:---|:---|:---|
|num<BR>(required)|int |Number of evaluation data. |
|accuracy<BR>(required) |double |Accuracy. |
|precision<BR>(required) |double[*N*][*M*] |Precision. |
|recall<BR>(required) |double[*N*][*M*] |Recall. |
|fvalue<BR>(required) |double[*N*][*M*] |F1 value. |

### server.py
This is the gRPC server boot script. Example code is available [here](server.py).

```python
import sys
import os
import pathlib


root_path = pathlib.Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(root_path))


from concurrent import futures
import grpc
import time

from rekcurd import RekcurdDashboardServicer, RekcurdWorkerServicer
from rekcurd.logger import JsonSystemLogger, JsonServiceLogger
from rekcurd.protobuf import rekcurd_pb2_grpc
from app import MyApp

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def serve():
    app = MyApp("./settings.yml")
    system_logger = JsonSystemLogger(app.config)
    service_logger = JsonServiceLogger(app.config)
    system_logger.info("Wake-up rekcurd worker.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    rekcurd_pb2_grpc.add_RekcurdDashboardServicer_to_server(
        RekcurdDashboardServicer(logger=system_logger, app=app), server)
    rekcurd_pb2_grpc.add_RekcurdWorkerServicer_to_server(
        RekcurdWorkerServicer(logger=service_logger, app=app), server)
    server.add_insecure_port("[::]:{0}".format(app.config.SERVICE_PORT))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        system_logger.info("Shutdown rekcurd worker.")
        server.stop(0)


if __name__ == '__main__':
    serve()
```

### start.sh
This is the boot script of your ML application. Example code is available [here](start.sh).

```bash
ECHO_PREFIX="[rekcurd example]: "

set -e
set -u

echo "$ECHO_PREFIX Start.."

pip install -r requirements.txt
python server.py
```

### logger.py (if necessary)
If you want to customize the logger, implement the interface class of [logger_interface.py](https://github.com/rekcurd/rekcurd-python/blob/master/rekcurd/logger/logger_interface.py)


## Try it!
```bash
$ sh start.sh
```
