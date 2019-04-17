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

#### `evaluate` method
Implement your ML model evaluating method. Argument `predictor` is your ML predictor defined by `load_model()`. Argument `filepath` is the file path of your evaluation data, and its format is your favorite.

### logger.py (if necessary)
If you want to customize the logger, implement the interface class of [logger_interface.py](https://github.com/rekcurd/rekcurd-python/blob/master/rekcurd/logger/logger_interface.py)

### start.sh
This is the boot script of your ML application. Example code is available [here](start.sh).


## Try it!
```bash
$ sh start.sh
```
