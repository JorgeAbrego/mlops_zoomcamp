import os
import pickle
import mlflow
import mlflow.sklearn

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, lr = data

    TRACKING_SERVER_HOST = "mlflow"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run():
        mlflow.set_tag("developer", "chuks")
        mlflow.set_tag("model", "linearRegression") 
        mlflow.set_tag("logtype", "auto") 
        with open("encoder.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("encoder.b", artifact_path="encoder")
        mlflow.sklearn.log_model(lr, artifact_path="linear_regression_models")

    return data