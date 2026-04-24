

import numpy as np
import pandas as pd


def _parabolic_sar(high: pd.Series, low: pd.Series, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Compute a simple Parabolic SAR series."""
    n = len(high)
    if n == 0:
        return pd.Series(dtype=float)

    high_vals = high.to_numpy(dtype=float)
    low_vals = low.to_numpy(dtype=float)
    sar = np.zeros(n, dtype=float)

    uptrend = True
    ep = high_vals[0]
    af = af_step
    sar[0] = low_vals[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if uptrend:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low_vals[i - 1])
            if i > 1:
                sar[i] = min(sar[i], low_vals[i - 2])

            if low_vals[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low_vals[i]
                af = af_step
            else:
                if high_vals[i] > ep:
                    ep = high_vals[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high_vals[i - 1])
            if i > 1:
                sar[i] = max(sar[i], high_vals[i - 2])

            if high_vals[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high_vals[i]
                af = af_step
            else:
                if low_vals[i] < ep:
                    ep = low_vals[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=high.index, name="sar")


def _return_streak(ret: pd.Series) -> pd.Series:
    """Consecutive run length of return sign (+/-), resets at sign changes or zero."""
    signs = np.sign(ret.fillna(0.0).to_numpy(dtype=float))
    streak = np.zeros_like(signs, dtype=float)

    for i in range(1, len(signs)):
        if signs[i] == 0:
            streak[i] = 0
        elif signs[i] == signs[i - 1]:
            streak[i] = streak[i - 1] + signs[i]
        else:
            streak[i] = signs[i]

    return pd.Series(streak, index=ret.index, name="ret_streak")


def _direction_3class_target(values) -> np.ndarray:
    """Map raw returns to {0, 1, 2} = {Down, Flat, Up}."""
    array = np.asarray(values)
    unique_values = set(np.unique(array).tolist())
    if unique_values.issubset({0, 1, 2}):
        return array.astype(int)
    array = array.astype(float)
    categorized = np.where(array > 0.01, 1, np.where(array < -0.01, -1, 0))
    return (categorized + 1).astype(int)


def make_sparse_macro_f1_metric(num_classes: int = 3, name: str = "macro_f1"):
    """Create a stateful sparse macro F1 metric for Keras classification models."""
    import importlib

    keras = importlib.import_module("keras")
    tf = importlib.import_module("tensorflow")

    class SparseMacroF1(keras.metrics.Metric):
        def __init__(self, num_classes: int = num_classes, name: str = name, **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.confusion_matrix = self.add_weight(
                name="confusion_matrix",
                shape=(num_classes, num_classes),
                initializer="zeros",
                dtype=tf.float32,
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            del sample_weight
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
            batch_cm = tf.math.confusion_matrix(
                y_true,
                y_pred,
                num_classes=self.num_classes,
                dtype=tf.float32,
            )
            self.confusion_matrix.assign_add(batch_cm)

        def result(self):
            true_positives = tf.linalg.tensor_diag_part(self.confusion_matrix)
            predicted_positives = tf.reduce_sum(self.confusion_matrix, axis=0)
            actual_positives = tf.reduce_sum(self.confusion_matrix, axis=1)

            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
            f1 = 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
            return tf.reduce_mean(f1)

        def reset_state(self):
            self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    return SparseMacroF1(num_classes=num_classes, name=name)


def add_technical_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """Append common technical indicators used for stock direction modeling."""
    data = X_df.sort_index().copy()

    close_col = "Adj Close" if "Adj Close" in data.columns else "Close" if "Close" in data.columns else None
    if close_col is None:
        raise ValueError("X_df must contain 'Adj Close' or 'Close' to build technical features")

    close = data[close_col].astype(float)
    ret = close.pct_change()
    data["ret_1d"] = ret

    # Moving averages of returns
    data["ret_ma_5"] = ret.rolling(5).mean()
    data["ret_ma_20"] = ret.rolling(20).mean()
    data["ret_ma_50"] = ret.rolling(50).mean()

    # MACD on price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # Volatility and Bollinger Bands
    price_ma20 = close.rolling(20).mean()
    price_std20 = close.rolling(20).std()
    data["volatility_20"] = ret.rolling(20).std()
    data["bb_mid"] = price_ma20
    data["bb_upper"] = price_ma20 + 2.0 * price_std20
    data["bb_lower"] = price_ma20 - 2.0 * price_std20
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / (price_ma20.abs() + 1e-12)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Indicators requiring OHLC
    required_ohlc = {"Open", "High", "Low"}
    if required_ohlc.issubset(set(data.columns)):
        high = data["High"].astype(float)
        low = data["Low"].astype(float)
        open_ = data["Open"].astype(float)

        data["sar"] = _parabolic_sar(high=high, low=low)

        tp = (high + low + close) / 3.0
        tp_ma20 = tp.rolling(20).mean()
        md20 = (tp - tp_ma20).abs().rolling(20).mean()
        data["cci_20"] = (tp - tp_ma20) / (0.015 * (md20 + 1e-12))

        prev_close = close.shift(1)
        ar_num = (high - open_).rolling(26).sum()
        ar_den = (open_ - low).rolling(26).sum()
        data["strat_ar"] = 100.0 * ar_num / (ar_den + 1e-12)

        br_num = (high - prev_close).clip(lower=0.0).rolling(26).sum()
        br_den = (prev_close - low).clip(lower=0.0).rolling(26).sum()
        data["strat_br"] = 100.0 * br_num / (br_den + 1e-12)

    # Volume-based indicators
    if "Volume" in data.columns:
        volume = data["Volume"].astype(float)
        vol_mean20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std()
        data["volume_z_20"] = (volume - vol_mean20) / (vol_std20 + 1e-12)

        pvi = np.full(len(data), np.nan, dtype=float)
        if len(data) > 0:
            pvi[0] = 1000.0
        for i in range(1, len(data)):
            prev = pvi[i - 1]
            if np.isnan(prev):
                prev = 1000.0
            if volume.iloc[i] > volume.iloc[i - 1]:
                pvi[i] = prev * (1.0 + (ret.iloc[i] if np.isfinite(ret.iloc[i]) else 0.0))
            else:
                pvi[i] = prev
        data["pvi"] = pvi

    # Return sign and streak
    data["ret_sign"] = np.sign(ret)
    data["ret_streak"] = _return_streak(ret)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def window_time_split(
    data: pd.DataFrame,
    target: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    window_size: int = 252,
    target_mode: str = "raw",
    return_index: bool = False,
):
    """Align data/target, then build rolling windows and split in time order."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    target_mode = target_mode.lower().strip()
    if target_mode not in {"raw", "direction_3class", "direction3", "categorical"}:
        raise ValueError("target_mode must be one of {'raw', 'direction_3class', 'direction3', 'categorical'}")

    data = data.sort_index().copy()
    target = target.sort_index().copy()

    common_index = data.index.intersection(target.index)
    data = data.loc[common_index].dropna()
    target = target.loc[data.index].dropna()

    # Re-align after dropna
    common_index = data.index.intersection(target.index)
    data = data.loc[common_index]
    target = target.loc[common_index]

    if len(data) != len(target):
        raise ValueError("data and target could not be aligned to the same length")

    if target_mode != "raw":
        target = pd.Series(_direction_3class_target(target.to_numpy()), index=target.index, name=target.name)

    X_raw = data.to_numpy()
    y_raw = target.to_numpy()
    idx_raw = data.index.to_numpy()

    n = len(data)
    if n <= window_size:
        raise ValueError(f"Not enough rows ({n}) for window_size={window_size}")

    # 1) Window first
    X_list, y_list, t_list = [], [], []
    for t in range(window_size, n):
        X_list.append(X_raw[t - window_size:t])  # past window
        y_list.append(y_raw[t])                  # predict current time t
        t_list.append(idx_raw[t])                # timestamp of label

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    t_idx = np.asarray(t_list)

    # 2) Split by time order
    total = len(X)
    n_train = int(total * train_size)
    n_val = int(total * val_size)

    train_end = n_train
    val_end = n_train + n_val

    x_train, y_train = X[:train_end], y[:train_end]
    x_val, y_val = X[train_end:val_end], y[train_end:val_end]
    x_test, y_test = X[val_end:], y[val_end:]

    if return_index:
        idx_train = t_idx[:train_end]
        idx_val = t_idx[train_end:val_end]
        idx_test = t_idx[val_end:]
        return x_train, y_train, x_val, y_val, x_test, y_test, idx_train, idx_val, idx_test

    return x_train, y_train, x_val, y_val, x_test, y_test

def single_stock_split(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    window_size: int = 252,
    add_transformed_features: int | bool = 1,
    target_mode: str = "raw",
    return_index: bool = False,
):
    """
    Build sliding windows first, then split by time order.
    X_df: feature dataframe, indexed by time
    y_series: target series, indexed by time

    add_transformed_features:
    - 0/False: no transformed features
    - non-zero/True: add transformed features and keep all original columns
    """
    data = X_df.sort_index().copy()

    # Keep compatibility with old int flags: 0/False means no add, any non-zero means add.
    add_flag = bool(add_transformed_features)
    if add_flag:
        data = add_technical_features(data)

    return window_time_split(
        data=data,
        target=y_series,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        window_size=window_size,
        target_mode=target_mode,
        return_index=return_index,
    )


def multi_stock_split(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    other_stocks_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    window_size: int = 252,
    add_transformed_features: int | bool = 1,
    target_mode: str = "raw",
    return_index: bool = False,
):
    """
    Build multi-stock features by:
    1) applying feature engineering on the main stock first,
    2) merging with other stock table by time index,
    3) then applying rolling window + time split.

    Parameters
    ----------
    X_df:
        Main stock feature dataframe (e.g., AAPL), indexed by time or with a 'Dt' column.
    y_series:
        Main stock target series indexed by time.
    other_stocks_df:
        Other stock wide table containing time-aligned features. It can be indexed by time or
        include a 'Dt' column.
    add_transformed_features:
        - 0/False: no transformed features on main stock
        - non-zero/True: add transformed features and keep all raw columns
    """
    data = X_df.sort_index().copy()

    # Keep compatibility with old int flags: 0/False means no add, any non-zero means add.
    add_flag = bool(add_transformed_features)
    if add_flag:
        data = add_technical_features(data)

    # Normalize time index for other stock table
    other = other_stocks_df.copy()
    if "Dt" in other.columns:
        other["Dt"] = pd.to_datetime(other["Dt"])
        other = other.sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
    else:
        other = other.sort_index().copy()
        try:
            other.index = pd.to_datetime(other.index)
        except Exception:
            pass

    # Normalize time index for main stock table
    if "Dt" in data.columns:
        data["Dt"] = pd.to_datetime(data["Dt"])
        data = data.sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
    else:
        data = data.sort_index().copy()
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            pass

    merged_data = data.join(other, how="inner")

    return window_time_split(
        data=merged_data,
        target=y_series,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        window_size=window_size,
        target_mode=target_mode,
        return_index=return_index,
    )


def evaluate_keras_model_on_validation(
    model,
    history,
    x_val: np.ndarray,
    y_val_cat: np.ndarray,
    prefix: str = "model",
    class_names: tuple[str, str, str] = ("Down", "Flat", "Up"),
    plot_history: bool = True,
):
    """Evaluate a trained Keras classification model on a validation set."""
    import importlib

    tf = importlib.import_module("tensorflow")
    keras = importlib.import_module("keras")
    metrics = importlib.import_module("sklearn.metrics")
    plt = importlib.import_module("matplotlib.pyplot")

    y_val_rnn = np.asarray(y_val_cat)
    unique_values = set(np.unique(y_val_rnn).tolist())
    if unique_values.issubset({0, 1, 2}):
        y_val_rnn = y_val_rnn.astype(int)
    elif unique_values.issubset({-1, 0, 1}):
        y_val_rnn = y_val_rnn.astype(int) + 1
    else:
        y_val_rnn = _direction_3class_target(y_val_rnn)

    eval_results = model.evaluate(x_val, y_val_rnn, verbose=0, return_dict=True)
    val_loss = float(eval_results["loss"])
    val_macro_f1 = float(eval_results.get("macro_f1", eval_results.get("accuracy", float("nan"))))
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print(f"[{prefix}] Validation macro_f1: {val_macro_f1:.4f}")
    print(f"[{prefix}] Validation loss: {val_loss:.4f}")
    print(f"\n[{prefix}] Validation classification report:")
    print(metrics.classification_report(
        y_val_rnn,
        y_val_pred,
        labels=[0, 1, 2],
        target_names=list(class_names),
        zero_division=0,
    ))
    print(f"\n[{prefix}] Validation confusion matrix:")
    print(metrics.confusion_matrix(y_val_rnn, y_val_pred, labels=[0, 1, 2]))

    if plot_history and history is not None:
        metric_key = "macro_f1" if "macro_f1" in history.history else "accuracy"
        val_metric_key = f"val_{metric_key}"

        plt.figure(figsize=(10, 4))
        plt.plot(history.history[metric_key], label=f"Train {metric_key}")
        plt.plot(history.history[val_metric_key], label=f"Val {metric_key}")
        plt.title(f"RNN {prefix} Training {metric_key}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_key)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title(f"RNN {prefix} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "val_loss": val_loss,
        "val_macro_f1": val_macro_f1,
        "y_val_pred": y_val_pred,
    }


