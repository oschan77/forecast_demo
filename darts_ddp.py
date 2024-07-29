import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import *
from darts.metrics import *
from darts.models import *
from darts.models import GlobalNaiveAggregate, GlobalNaiveDrift, GlobalNaiveSeasonal
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def forecast_and_plot(model, series_train, series_val, n_jobs=True, plot=False):
    if n_jobs:
        pred = model.predict(n=len(series_val), series=series_train, n_jobs=-1)
    else:
        pred = model.predict(n=len(series_val), series=series_train)

    mape_loss = mape(series_val, pred)
    mae_loss = mae(series_val, pred)

    if plot:
        fig, ax = plt.subplots(figsize=(30, 5))
        series_train.plot(label="train")
        series_val.plot(label="true")
        pred.plot(label="prediction")

    return mape_loss, mae_loss


def load_all_timeseries(dir_path):
    all_files = []
    all_series = []
    train_series = []
    val_series = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            ts = TimeSeries.from_csv(
                file_path,
                time_col="Date",
                value_cols="Weekly_Sales",
                freq="W-FRI",
            )
            train_ts, val_ts = ts.split_before(0.70)

            all_files.append(file)
            all_series.append(ts)
            train_series.append(train_ts)
            val_series.append(val_ts)

    return all_files, all_series, train_series, val_series


def eval_on_all_series(model, train_series, val_series, n_jobs=True):
    mape_losses, mae_losses = [], []
    for i in range(len(val_series)):
        mape_loss, mae_loss = forecast_and_plot(
            model=model,
            series_train=train_series[i],
            series_val=val_series[i],
            n_jobs=n_jobs,
            plot=False,
        )
        mape_losses.append(mape_loss)
        mae_losses.append(mae_loss)

    return mape_losses, mae_losses


def print_losses(model_name, mape_losses, mae_losses):
    print(f"{model_name}:")
    print(f"mean MAPE: {np.mean(mape_losses)}")
    print(f"mean MAE: {np.mean(mae_losses)}")


if __name__ == "__main__":
    all_files, all_series, train_series, val_series = load_all_timeseries(
        "data/processed/walmart_sales/BySD500"
    )

    early_stopping = EarlyStopping(
        monitor="train_loss",
        patience=20,
        min_delta=1e2,
        mode="min",
    )

    nhits_model = NHiTSModel(
        input_chunk_length=64,
        output_chunk_length=12,
        num_stacks=5,
        num_blocks=2,
        num_layers=3,
        dropout=0.2,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": -1,
            "strategy": "ddp",
            "callbacks": [early_stopping],
        },
    )

    nhits_model.fit(train_series, epochs=500, verbose=True)

    nhits_mape_losses, nhits_mae_losses = eval_on_all_series(
        nhits_model, train_series, val_series
    )

    print_losses("nhits_model", nhits_mape_losses, nhits_mae_losses)

    linear_regression_model = LinearRegressionModel(
        lags=64,
        output_chunk_length=12,
    )

    linear_regression_model.fit(train_series)

    linear_regression_mape_losses, linear_regression_mae_losses = eval_on_all_series(
        linear_regression_model,
        train_series,
        val_series,
        n_jobs=False,
    )

    print_losses(
        "linear_regression_model",
        linear_regression_mape_losses,
        linear_regression_mae_losses,
    )
