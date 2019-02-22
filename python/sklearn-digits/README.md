# Rekcurd-example python-sklearn-digits
This is the example of [Rekcurd](https://github.com/rekcurd/rekcurd-python).

## How to use it
### settings.yml
This is the configuration file of your ML module. Template is available [here](settings.yml).

### app.py
This is the Rekcurdized application of your ML module. Example code is available [here](app.py). You must implement the following methods.

- `load_method`
- `predict`
- `evaluate`
- `get_evaluate_detail`

After that you can run your application by following.
```python
if __name__ == '__main__':
    app = MyApp()
    app.load_config_file("./settings.yml")
    app.run()
```

#### `load_model` method
Implement your ML model loading method. Argument `filepath` is the file path of your ML model. You can use a compressed file (e.g. zip) if you want.

**Example**
```python
def joblib_load_from_zip(self, zip_name: str, file_name: str):
    with zipfile.ZipFile(zip_name, 'r') as zf:
        with zf.open(file_name, 'r') as zipmodel:
            return joblib.load(io.BufferedReader(io.BytesIO(zipmodel.read())))

def load_model(self, filepath: str) -> object:
    try:
        file_name = 'default.model'
        return self.joblib_load_from_zip(filepath, file_name)
    except Exception as e:
        """Stop gRPC server."""
        os._exit(-1)
```

#### `predict` method
Implement your ML model predicting/inferring method. Argument `predictor` is your ML predictor defined by `load_model()`.

**Definition**
```python
def predict(self, predictor: object, idata: PredictInput, option: dict = None) -> PredictResult:
```

##### `PredictLabel` type
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

##### `PredictResult` type
*M* is the number of classes. If your algorithm is a binary classifier, you set *M* to 1. If your algorithm is a multi-class classifier, you set *M* to the number of classes.

|Field |Type |Description |
|:---|:---|:---|
|label<BR>(required) |One of below<BR> -string<BR> -bytes<BR> -string[*M*]<BR> -int[*M*]<BR> -double[*M*] |Result of inference.<BR> -"positive" for a sentiment analysis.<BR> -PNG file for an image transformation.<BR> -["a", "b"] for a multi-class classification.<BR> -[1, 2] for a multi-class classification.<BR> -[0.9, 0.1] for a multi-class classification. |
|score<BR>(required) |One of below<BR> -double<BR> -double[*M*] |Score of result.<BR> -0.98 for a binary classification.<BR> -[0.9, 0.1] for a multi-class classification. |
|option |string |Option field. Must be json format. |

#### `evaluate` method
Implement your ML model evaluating method. Argument `predictor` is your ML predictor defined by `load_model()`. Argument `filepath` is the file path of your evaluation data, and its format is your favorite.

**Definition**
```python
def evaluate(self, predictor: object, filepath: str) -> Tuple[EvaluateResult, List[EvaluateResultDetail]]:
```

##### `EvaluateResult` type
`EvaluateResult` is the evaluation score. *N* is the number of evaluation data. *M* is the number of classes. If your algorithm is a binary classifier, you set *M* to 1. If your algorithm is a multi-class classifier, you set *M* to the number of classes.

|Field |Type |Description |
|:---|:---|:---|
|num<BR>(required)|int |Number of evaluation data. |
|accuracy<BR>(required) |double |Accuracy. |
|precision<BR>(required) |double[*N*][*M*] |Precision. |
|recall<BR>(required) |double[*N*][*M*] |Recall. |
|fvalue<BR>(required) |double[*N*][*M*] |F1 value. |

##### `EvaluateDetail` type
`EvaluateDetail` is the details of evaluation result.

|Field |Type |Description |
|:---|:---|:---|
|result<BR>(required) |PredictResult |Prediction result. |
|is_correct<BR>(required) |bool |Correct or not. |


### logger.py (if necessary)
If you want to customize the logger, implement the interface class of [logger_interface.py](https://github.com/rekcurd/rekcurd-python/blob/master/rekcurd/logger/logger_interface.py)

### start.sh
This is the boot script of your ML application. Example code is available [here](start.sh).


## Try it!
```bash
$ sh start.sh
```
