

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations


def set_global_determinism(seed: int = 42, deterministic_ops: bool = True) -> None:
    """Seed Python, NumPy, Keras, and TensorFlow before building a model."""
    import importlib
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        tf = importlib.import_module("tensorflow")
    except ImportError:
        tf = None

    if tf is not None:
        if hasattr(tf.keras.utils, "set_random_seed"):
            tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass

    try:
        keras = importlib.import_module("keras")
    except ImportError:
        keras = None

    if keras is not None and hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)


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


def _direction_binary_target(values) -> np.ndarray:
    """Map raw returns to {0, 1} = {Down, Up} using zero as the boundary."""
    array = np.asarray(values)
    unique_values = set(np.unique(array).tolist())

    if np.issubdtype(array.dtype, np.integer) and unique_values.issubset({0, 1}):
        return array.astype(int)

    if unique_values.issubset({0, 1, 2}) and 2 in unique_values:
        target = np.full(array.shape, -1, dtype=int)
        target[array == 0] = 0
        target[array == 2] = 1
        return target

    if unique_values.issubset({-1, 0, 1}):
        target = np.full(array.shape, -1, dtype=int)
        target[array == -1] = 0
        target[array == 1] = 1
        return target

    array = array.astype(float)
    return np.where(array > 0.0, 1, 0).astype(int)


def _direction_binary_filtered_target(values, threshold: float = 0.01) -> np.ndarray:
    """Map raw returns to {-1, 0, 1}; -1 is an ambiguous Flat sample to drop."""
    array = np.asarray(values)
    unique_values = set(np.unique(array).tolist())

    if np.issubdtype(array.dtype, np.integer) and unique_values.issubset({0, 1}):
        return array.astype(int)

    if unique_values.issubset({0, 1, 2}) and 2 in unique_values:
        target = np.full(array.shape, -1, dtype=int)
        target[array == 0] = 0
        target[array == 2] = 1
        return target

    if unique_values.issubset({-1, 0, 1}):
        target = np.full(array.shape, -1, dtype=int)
        target[array == -1] = 0
        target[array == 1] = 1
        return target

    array = array.astype(float)
    return np.where(array > threshold, 1, np.where(array < -threshold, 0, -1)).astype(int)


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

        # Compute SAR only on rows with valid OHLC to avoid full-column NaN
        # when a stock has a long pre-listing missing segment.
        valid_ohlc = high.notna() & low.notna() & open_.notna() & close.notna()
        sar = pd.Series(np.nan, index=data.index, dtype=float)
        if valid_ohlc.any():
            sar_valid = _parabolic_sar(high=high.loc[valid_ohlc], low=low.loc[valid_ohlc])
            sar.loc[sar_valid.index] = sar_valid
        data["sar"] = sar

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


def _apply_transformed_feature_mode(data: pd.DataFrame, add_transformed_features: int | bool) -> pd.DataFrame:
    """Apply the feature flag used by split helpers.

    0/False: keep original columns only.
    1/True: keep original columns plus engineered technical features.
    2: keep only engineered technical features.
    """
    if isinstance(add_transformed_features, bool):
        mode = int(add_transformed_features)
    else:
        mode = int(add_transformed_features)

    if mode == 0:
        return data.sort_index().copy()
    if mode not in {1, 2}:
        raise ValueError("add_transformed_features must be one of {0, 1, 2} or a boolean")

    original_cols = list(data.columns)
    transformed = add_technical_features(data)
    if mode == 1:
        return transformed

    engineered_cols = [col for col in transformed.columns if col not in original_cols]
    if not engineered_cols:
        raise ValueError("add_transformed_features=2 produced no engineered feature columns")
    return transformed.loc[:, engineered_cols]


def add_rolling_alpha_beta_from_ret1d(
    stock_features_df: pd.DataFrame,
    market_data: pd.DataFrame | str | Path,
    rolling_window: int = 252,
) -> pd.DataFrame:
    """Append rolling alpha/beta features using existing ret_1d from technical-feature outputs.

    This function is designed to be called after ``add_technical_features`` so the
    stock return series comes directly from ``ret_1d`` instead of recomputing pct_change.
    """
    if rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")

    stock = stock_features_df.sort_index().copy()
    if "ret_1d" not in stock.columns:
        raise ValueError("stock_features_df must contain 'ret_1d'. Run add_technical_features first.")

    if isinstance(market_data, (str, Path)):
        market_raw = pd.read_csv(market_data)
    elif isinstance(market_data, pd.DataFrame):
        market_raw = market_data.copy()
    else:
        raise TypeError("market_data must be a pandas DataFrame, str path, or pathlib.Path")

    if "Dt" in market_raw.columns:
        market_raw["Dt"] = pd.to_datetime(market_raw["Dt"], format="%Y-%m-%d", errors="coerce")
        market_raw = market_raw.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
    else:
        market_raw = market_raw.sort_index().copy()
        market_raw.index = pd.to_datetime(market_raw.index, errors="coerce")
        market_raw = market_raw[market_raw.index.notna()]

    market_features = add_technical_features(market_raw)
    if "ret_1d" not in market_features.columns:
        raise ValueError("market_data could not produce 'ret_1d'.")

    pair_ret = pd.concat(
        [
            stock["ret_1d"].rename("stock_ret"),
            market_features["ret_1d"].rename("market_ret"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    rolling_cov = pair_ret["stock_ret"].rolling(rolling_window).cov(pair_ret["market_ret"])
    rolling_var = pair_ret["market_ret"].rolling(rolling_window).var()
    beta_raw = rolling_cov / (rolling_var + 1e-12)

    mean_stock = pair_ret["stock_ret"].rolling(rolling_window).mean()
    mean_market = pair_ret["market_ret"].rolling(rolling_window).mean()
    alpha_raw = mean_stock - beta_raw * mean_market

    beta_col = f"beta_{rolling_window}"
    alpha_col = f"alpha_{rolling_window}"

    pair_ret[beta_col] = beta_raw.shift(1)
    pair_ret[alpha_col] = alpha_raw.shift(1)

    stock = stock.join(pair_ret[[beta_col, alpha_col]], how="left")
    stock = stock.replace([np.inf, -np.inf], np.nan)
    return stock

def build_market_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    spy = add_technical_features(spy_df).add_prefix("spy_")

    keep_cols = [
        "spy_ret_1d",
        "spy_ret_ma_5",
        "spy_ret_ma_20",
        "spy_ret_ma_50",
        "spy_volatility_20",
        "spy_macd",
        "spy_macd_signal",
        "spy_macd_hist",
        "spy_bb_width",
        "spy_rsi_14",
        "spy_volume_z_20",
        "spy_ret_sign",
        "spy_ret_streak",
    ]

    keep_cols = [c for c in keep_cols if c in spy.columns]
    return spy[keep_cols]


def add_more_market_feature(
    stock_features_df: pd.DataFrame,
    market_data: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Append higher-level market features to an already-expanded stock table.

    Existing columns are preserved. The helper only adds columns that are missing.
    It is intended to be used at the end of the notebook after the core feature
    engineering has already been done.
    """

    def _normalize_time_index(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if "Dt" in normalized.columns:
            normalized["Dt"] = pd.to_datetime(normalized["Dt"], format="%Y-%m-%d", errors="coerce")
            normalized = normalized.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
        else:
            normalized = normalized.sort_index().copy()
            normalized.index = pd.to_datetime(normalized.index, errors="coerce")
            normalized = normalized[normalized.index.notna()]
        return normalized

    def _set_if_missing(frame: pd.DataFrame, column_name: str, values) -> None:
        if column_name not in frame.columns:
            frame[column_name] = values

    stock = stock_features_df.sort_index().copy()
    if "ret_1d" not in stock.columns:
        stock = add_technical_features(stock)

    if isinstance(market_data, (str, Path)):
        market_raw = pd.read_csv(market_data)
    elif isinstance(market_data, pd.DataFrame):
        market_raw = market_data.copy()
    else:
        raise TypeError("market_data must be a pandas DataFrame, str path, or pathlib.Path")

    market_raw = _normalize_time_index(market_raw)

    # Use the standard market feature subset as a baseline, but do not overwrite anything.
    market_features = build_market_features(market_raw).reindex(stock.index)
    for column_name in market_features.columns:
        _set_if_missing(stock, column_name, market_features[column_name])

    market_close_col = "Adj Close" if "Adj Close" in market_raw.columns else "Close" if "Close" in market_raw.columns else None
    if market_close_col is None:
        return stock.replace([np.inf, -np.inf], np.nan)

    stock_ret = stock["ret_1d"].astype(float)
    market_ret = add_technical_features(market_raw)["ret_1d"].astype(float).reindex(stock.index)
    pair_ret = pd.concat(
        [stock_ret.rename("stock_ret"), market_ret.rename("market_ret")],
        axis=1,
        join="inner",
    ).dropna()

    # Volatility regime.
    _set_if_missing(stock, "spy_volatility_60", market_ret.rolling(60).std().shift(1).reindex(stock.index))
    if "spy_volatility_20" in stock.columns and "spy_volatility_60" in stock.columns:
        _set_if_missing(
            stock,
            "spy_volatility_ratio_20_60",
            stock["spy_volatility_20"] / (stock["spy_volatility_60"] + 1e-12),
        )

    # Trend slope / acceleration.
    if {"spy_ret_ma_5", "spy_ret_ma_20"}.issubset(stock.columns):
        _set_if_missing(stock, "spy_trend_5_20", stock["spy_ret_ma_5"] - stock["spy_ret_ma_20"])
    if {"spy_ret_ma_20", "spy_ret_ma_50"}.issubset(stock.columns):
        _set_if_missing(stock, "spy_trend_20_50", stock["spy_ret_ma_20"] - stock["spy_ret_ma_50"])

    # Dynamic beta / alpha.
    for window in (20, 60, 252):
        beta_col = f"beta_{window}"
        alpha_col = f"alpha_{window}"
        rolling_cov = pair_ret["stock_ret"].rolling(window).cov(pair_ret["market_ret"])
        rolling_var = pair_ret["market_ret"].rolling(window).var()
        beta_raw = rolling_cov / (rolling_var + 1e-12)
        mean_stock = pair_ret["stock_ret"].rolling(window).mean()
        mean_market = pair_ret["market_ret"].rolling(window).mean()
        alpha_raw = mean_stock - beta_raw * mean_market
        _set_if_missing(stock, beta_col, beta_raw.shift(1).reindex(stock.index))
        _set_if_missing(stock, alpha_col, alpha_raw.shift(1).reindex(stock.index))

    if {"beta_20", "beta_252"}.issubset(stock.columns):
        _set_if_missing(stock, "beta_20_minus_252", stock["beta_20"] - stock["beta_252"])
    if {"alpha_20", "alpha_60"}.issubset(stock.columns):
        _set_if_missing(stock, "alpha_20_minus_60", stock["alpha_20"] - stock["alpha_60"])

    # SPY shock flag.
    shock_threshold = market_ret.abs().rolling(60).quantile(0.95).shift(1)
    _set_if_missing(stock, "spy_shock_flag_q95_60", (market_ret.abs() > shock_threshold).astype(float).reindex(stock.index))

    # Direction agreement and lagged correlation.
    sign_match = (np.sign(stock_ret.fillna(0.0)) == np.sign(market_ret.fillna(0.0))).astype(float)
    for window in (20, 60):
        _set_if_missing(stock, f"aapl_spy_sign_agree_{window}", sign_match.rolling(window).mean().shift(1).reindex(stock.index))
        _set_if_missing(stock, f"aapl_spy_lag_corr_{window}", stock_ret.rolling(window).corr(market_ret.shift(1)).shift(1).reindex(stock.index))

    # SPY volume / price-pressure interactions.
    if {"spy_volume_z_20", "spy_ret_sign"}.issubset(stock.columns):
        _set_if_missing(stock, "spy_volume_pressure_20", stock["spy_volume_z_20"] * stock["spy_ret_sign"])
    if {"spy_volume_z_20", "spy_ret_ma_20"}.issubset(stock.columns):
        _set_if_missing(stock, "spy_volume_momentum_pressure_20", stock["spy_volume_z_20"] * stock["spy_ret_ma_20"])
    if {"spy_bb_width", "spy_macd_hist"}.issubset(stock.columns):
        _set_if_missing(stock, "spy_bb_macd_pressure", stock["spy_bb_width"] * stock["spy_macd_hist"])

    # Relative strength factor.
    stock_close_col = "Adj Close" if "Adj Close" in stock.columns else "Close" if "Close" in stock.columns else None
    if stock_close_col is not None:
        price_pair = pd.concat(
            [
                stock[stock_close_col].astype(float).rename("stock_close"),
                market_raw[market_close_col].astype(float).rename("market_close"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        price_ratio = np.log(price_pair["stock_close"] / price_pair["market_close"])
        _set_if_missing(stock, "aapl_spy_log_price_ratio", price_ratio.reindex(stock.index))
        _set_if_missing(stock, "aapl_spy_log_price_ratio_ma_20", price_ratio.rolling(20).mean().shift(1).reindex(stock.index))
        _set_if_missing(stock, "aapl_spy_log_price_ratio_slope_20", (price_ratio - price_ratio.shift(20)).shift(1).reindex(stock.index))

    # Simple market-state bucket.
    if {"spy_ret_ma_20", "spy_volatility_20"}.issubset(stock.columns):
        momentum = stock["spy_ret_ma_20"] - stock.get("spy_ret_ma_50", 0.0)
        volatility_ref = stock["spy_volatility_20"].rolling(60).median().shift(1)
        market_state = np.where(
            (momentum > 0) & (stock["spy_volatility_20"] <= volatility_ref),
            2.0,
            np.where((momentum < 0) & (stock["spy_volatility_20"] > volatility_ref), 0.0, 1.0),
        )
        _set_if_missing(stock, "spy_market_state", pd.Series(market_state, index=stock.index).where(volatility_ref.notna(), 1.0))

    # Keep the original cross-asset features if they were not already created.
    _set_if_missing(stock, "aapl_excess_ret_1d", stock["ret_1d"] - stock.get("spy_ret_1d", stock["ret_1d"]))
    if {"ret_ma_5", "spy_ret_ma_5"}.issubset(stock.columns):
        _set_if_missing(stock, "aapl_excess_ret_ma_5", stock["ret_ma_5"] - stock["spy_ret_ma_5"])
    if {"ret_ma_20", "spy_ret_ma_20"}.issubset(stock.columns):
        _set_if_missing(stock, "aapl_excess_ret_ma_20", stock["ret_ma_20"] - stock["spy_ret_ma_20"])

    return stock.replace([np.inf, -np.inf], np.nan)


def build_plus_sector_aapl_excess_and_volume_features(
    aapl_data: pd.DataFrame,
    data_dir: str | Path,
    sector_tickers: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the selected feature baseline used after v6 feature testing.

    The returned table corresponds to ``plus_sector_aapl_excess_and_volume``:
    AAPL technical features, SPY market/excess features without alpha/beta,
    extra non-alpha/beta SPY market features, and sector ETF features measured
    directly against AAPL. Other individual company features are intentionally
    excluded because they did not improve validation performance in the notebook.
    """

    data_dir = Path(data_dir)
    spy_path = data_dir / "SPY.csv"
    if sector_tickers is None:
        sector_tickers = ("XLU", "XLP", "XLK", "XLI", "XLF", "XLE", "XLB", "XLV")

    def _normalize_price_volume_table(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if "Dt" in normalized.columns:
            normalized["Dt"] = pd.to_datetime(normalized["Dt"], format="%Y-%m-%d", errors="coerce")
            normalized = normalized.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
        else:
            normalized = normalized.sort_index().copy()
            normalized.index = pd.to_datetime(normalized.index, errors="coerce")
            normalized = normalized[normalized.index.notna()]
        return normalized

    def _load_price_volume_table(ticker: str) -> pd.DataFrame:
        path = data_dir / f"{ticker}.csv"
        raw = pd.read_csv(path)
        return _normalize_price_volume_table(raw)

    def _rolling_volume_z(volume: pd.Series, window: int = 20) -> pd.Series:
        volume = volume.astype(float)
        return (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-12)

    aapl_raw = _normalize_price_volume_table(aapl_data)
    aapl_base_features = add_technical_features(aapl_raw)
    aapl_raw_cols = [col for col in aapl_raw.columns if col in aapl_base_features.columns]
    aapl_base_features = aapl_base_features.drop(columns=aapl_raw_cols)

    spy_raw = _load_price_volume_table("SPY")
    spy_features = build_market_features(spy_raw).reindex(aapl_base_features.index)

    excess_features = pd.DataFrame(index=aapl_base_features.index)
    excess_features["aapl_excess_ret_1d"] = aapl_base_features["ret_1d"] - spy_features["spy_ret_1d"]
    excess_features["aapl_excess_ret_ma_5"] = aapl_base_features["ret_ma_5"] - spy_features["spy_ret_ma_5"]
    excess_features["aapl_excess_ret_ma_20"] = aapl_base_features["ret_ma_20"] - spy_features["spy_ret_ma_20"]

    no_alpha_beta_baseline = pd.concat(
        [aapl_base_features, spy_features, excess_features],
        axis=1,
    )
    no_alpha_beta_baseline = no_alpha_beta_baseline.loc[:, ~no_alpha_beta_baseline.columns.duplicated()]

    extra_market_candidate = add_more_market_feature(no_alpha_beta_baseline.copy(), spy_path)
    extra_market_cols = [
        col for col in extra_market_candidate.columns
        if col not in no_alpha_beta_baseline.columns
        and col.startswith(("spy_", "aapl_spy_", "aapl_excess_"))
        and not col.startswith(("alpha_", "beta_"))
    ]
    skipped_alpha_beta_cols = [
        col for col in extra_market_candidate.columns
        if col not in no_alpha_beta_baseline.columns
        and col.startswith(("alpha_", "beta_"))
    ]
    extra_market_features = extra_market_candidate.loc[:, extra_market_cols]

    extra_market_baseline = pd.concat(
        [no_alpha_beta_baseline, extra_market_features],
        axis=1,
    )
    extra_market_baseline = extra_market_baseline.loc[:, ~extra_market_baseline.columns.duplicated()]

    aapl_features_for_sector = add_technical_features(aapl_raw)
    aapl_ret_1d = aapl_features_for_sector["ret_1d"]
    aapl_ret_ma_5 = aapl_features_for_sector["ret_ma_5"]
    aapl_ret_ma_20 = aapl_features_for_sector["ret_ma_20"]
    aapl_volume = aapl_raw["Volume"].astype(float)
    aapl_volume_chg_1d = aapl_volume.pct_change()
    aapl_volume_z_20 = _rolling_volume_z(aapl_volume, window=20)
    aapl_volume_ma_20 = aapl_volume.rolling(20).mean()

    sector_blocks = []
    sector_summary = []
    for ticker in sector_tickers:
        sector_raw = _load_price_volume_table(ticker)
        sector_features = add_technical_features(sector_raw)
        sector_volume = sector_raw["Volume"].astype(float)

        block = pd.DataFrame(index=extra_market_baseline.index)

        sector_excess_ret_1d = aapl_ret_1d - sector_features["ret_1d"]
        sector_excess_ret_ma_5 = aapl_ret_ma_5 - sector_features["ret_ma_5"]
        sector_excess_ret_ma_20 = aapl_ret_ma_20 - sector_features["ret_ma_20"]
        block[f"AAPL_{ticker}_excess_ret_1d"] = sector_excess_ret_1d.reindex(block.index)
        block[f"AAPL_{ticker}_excess_ret_ma_5"] = sector_excess_ret_ma_5.reindex(block.index)
        block[f"AAPL_{ticker}_excess_ret_ma_20"] = sector_excess_ret_ma_20.reindex(block.index)
        block[f"AAPL_{ticker}_excess_ret_mom_5_20"] = (sector_excess_ret_ma_5 - sector_excess_ret_ma_20).reindex(block.index)

        sector_volume_chg_1d = sector_volume.pct_change()
        sector_volume_z_20 = _rolling_volume_z(sector_volume, window=20)
        sector_volume_ma_20 = sector_volume.rolling(20).mean()
        block[f"{ticker}_AAPL_volume_chg_excess_1d"] = (sector_volume_chg_1d - aapl_volume_chg_1d).reindex(block.index)
        block[f"{ticker}_AAPL_volume_z_20_spread"] = (sector_volume_z_20 - aapl_volume_z_20).reindex(block.index)
        block[f"{ticker}_AAPL_volume_ratio_20"] = ((sector_volume_ma_20 / (aapl_volume_ma_20 + 1e-12)) - 1.0).reindex(block.index)

        sector_has_data = sector_raw.notna().any(axis=1)
        sector_first_available = sector_has_data[sector_has_data].index.min() if sector_has_data.any() else pd.NaT

        sector_blocks.append(block)
        sector_summary.append({
            "ticker": ticker,
            "first_available_date": sector_first_available,
            "last_date": sector_raw.index.max(),
            "features_added": block.shape[1],
        })

    sector_features = pd.concat(sector_blocks, axis=1).replace([np.inf, -np.inf], np.nan)

    selected_features = pd.concat(
        [extra_market_baseline, sector_features],
        axis=1,
    )
    selected_features = selected_features.loc[:, ~selected_features.columns.duplicated()]
    selected_features = selected_features.replace([np.inf, -np.inf], np.nan)

    audit = {
        "feature_set_name": "plus_sector_aapl_excess_and_volume",
        "aapl_raw_columns_dropped": aapl_raw_cols,
        "aapl_base_shape": aapl_base_features.shape,
        "spy_feature_shape": spy_features.shape,
        "excess_feature_shape": excess_features.shape,
        "no_alpha_beta_baseline_shape": no_alpha_beta_baseline.shape,
        "extra_market_columns": extra_market_cols,
        "extra_market_shape": extra_market_features.shape,
        "skipped_alpha_beta_columns": skipped_alpha_beta_cols,
        "extra_market_baseline_shape": extra_market_baseline.shape,
        "sector_tickers": list(sector_tickers),
        "sector_feature_columns": sector_features.columns.tolist(),
        "sector_feature_shape": sector_features.shape,
        "sector_summary": pd.DataFrame(sector_summary),
        "final_shape": selected_features.shape,
        "alpha_beta_columns_in_final": [
            col for col in selected_features.columns
            if col.startswith(("alpha_", "beta_"))
        ],
    }

    return selected_features, audit


def add_interaction_features(
    features_df: pd.DataFrame,
    interaction_degree: int = 2,
    top_k: int | None = 100,
) -> pd.DataFrame:
    """Append multiplicative interaction features from high-variance numeric columns.

    ``top_k`` controls how many original numeric columns are used to build
    interactions. For example, ``top_k=100`` and ``interaction_degree=2`` adds
    4,950 pairwise product features.
    """
    if interaction_degree < 2:
        raise ValueError("interaction_degree must be >= 2")
    if top_k is not None and top_k < 2:
        raise ValueError("top_k must be None or >= 2")

    data = features_df.copy()
    numeric_data = data.select_dtypes(include=[np.number]).astype(float)
    if numeric_data.shape[1] < 2:
        return data

    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    usable_cols = numeric_data.columns[numeric_data.notna().any(axis=0)].tolist()
    if len(usable_cols) < 2:
        return data

    col_std = numeric_data[usable_cols].std(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ranked_cols = col_std[col_std > 0.0].sort_values(ascending=False).index.tolist()
    if len(ranked_cols) < 2:
        return data

    if top_k is not None:
        ranked_cols = ranked_cols[: min(top_k, len(ranked_cols))]

    new_blocks = []
    existing_cols = set(data.columns)
    selected_numeric = numeric_data[ranked_cols]

    for degree in range(2, interaction_degree + 1):
        interaction_columns = {}
        for cols in combinations(ranked_cols, degree):
            col_name = "__x__".join(cols)
            if col_name in existing_cols:
                continue

            values = selected_numeric.loc[:, cols[0]]
            for col in cols[1:]:
                values = values * selected_numeric.loc[:, col]
            interaction_columns[col_name] = values
            existing_cols.add(col_name)

        if interaction_columns:
            new_blocks.append(pd.DataFrame(interaction_columns, index=data.index))

    if new_blocks:
        data = pd.concat([data, *new_blocks], axis=1)

    return data.replace([np.inf, -np.inf], np.nan)


def add_cross_asset_relation_features(expanded_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add cross-asset relational features on top of an expanded AAPL feature table.

    Expected inputs are columns from AAPL (unprefixed) plus prefixed columns from other
    symbols generated by ``expand_with_other_stock_features``.
    """
    df = expanded_df.copy()
    new_columns: dict[str, pd.Series] = {}
    added_cols: list[str] = []

    def _add_col(name: str, series: pd.Series) -> None:
        if name not in df.columns and name not in new_columns:
            new_columns[name] = series
            added_cols.append(name)

    if "ret_1d" not in df.columns:
        return df, added_cols

    aapl_ret = df["ret_1d"].astype(float)

    # 1) Breadth & dispersion among related tech leaders.
    peer_tickers = ["MSFT", "GOOG", "NVDA", "ADBE", "CRM", "CSCO"]
    peer_ret_cols = [f"{t}_ret_1d" for t in peer_tickers if f"{t}_ret_1d" in df.columns]
    if peer_ret_cols:
        peer_rets = df[peer_ret_cols].astype(float)
        _add_col("peer_breadth_up", (peer_rets > 0).mean(axis=1))
        _add_col("peer_dispersion", peer_rets.std(axis=1))

    # 2) Relative returns: AAPL vs key assets.
    for t in ["XLK", "MSFT", "SPY"]:
        col = f"{t}_ret_1d"
        if col in df.columns:
            _add_col(f"aapl_minus_{t.lower()}_ret_1d", aapl_ret - df[col].astype(float))

    # 3) Price-spread momentum (log price ratio momentum).
    aapl_close_col = "Adj Close" if "Adj Close" in df.columns else "Close" if "Close" in df.columns else None
    if aapl_close_col is not None:
        aapl_close = df[aapl_close_col].astype(float)
        for t in ["XLK", "MSFT", "SPY"]:
            peer_close_col = f"{t}_Adj Close" if f"{t}_Adj Close" in df.columns else f"{t}_Close" if f"{t}_Close" in df.columns else None
            if peer_close_col is None:
                continue
            log_ratio = np.log(aapl_close / df[peer_close_col].astype(float))
            _add_col(f"aapl_{t.lower()}_log_price_ratio", log_ratio)
            _add_col(f"aapl_{t.lower()}_spread_mom_5", (log_ratio - log_ratio.shift(5)).shift(1))
            _add_col(f"aapl_{t.lower()}_spread_mom_20", (log_ratio - log_ratio.shift(20)).shift(1))

    # 4) Rolling correlations (20/60): AAPL vs XLK/MSFT/SPY.
    for t in ["XLK", "MSFT", "SPY"]:
        col = f"{t}_ret_1d"
        if col not in df.columns:
            continue
        peer_ret = df[col].astype(float)
        _add_col(f"corr20_aapl_{t.lower()}", aapl_ret.rolling(20).corr(peer_ret).shift(1))
        _add_col(f"corr60_aapl_{t.lower()}", aapl_ret.rolling(60).corr(peer_ret).shift(1))

    # 5) Sector relative strength and rotation speed vs SPY.
    if "SPY_ret_1d" in df.columns:
        spy_ret = df["SPY_ret_1d"].astype(float)
        for t in ["XLK", "XLY", "XLF", "XLI", "XLV", "XLP", "XLE", "XLB", "XLU"]:
            col = f"{t}_ret_1d"
            if col in df.columns:
                rs_col = f"{t.lower()}_rel_spy_ret_1d"
                rs = df[col].astype(float) - spy_ret
                _add_col(rs_col, rs)
                _add_col(f"{t.lower()}_rel_spy_ret_5", rs.rolling(5).mean().shift(1))
                _add_col(f"{t.lower()}_rotation_speed_5", rs.diff(5).shift(1))

    # 6) Explicit lead-lag return features for key assets (priority list).
    lead_lag_tickers = ["SPY", "XLK", "MSFT", "GOOG", "NVDA"]
    for t in lead_lag_tickers:
        col = f"{t}_ret_1d"
        if col not in df.columns:
            continue
        ret_t = df[col].astype(float)
        for lag in (1, 3, 5):
            _add_col(f"{t.lower()}_ret_1d_lag{lag}", ret_t.shift(lag))

    # 7) Volatility change/regime features for key assets.
    for t in lead_lag_tickers:
        col = f"{t}_ret_1d"
        if col not in df.columns:
            continue
        ret_t = df[col].astype(float)
        vol20 = ret_t.rolling(20).std()
        vol60 = ret_t.rolling(60).std()
        _add_col(f"{t.lower()}_vol20", vol20)
        _add_col(f"{t.lower()}_vol60", vol60)
        _add_col(f"{t.lower()}_vol20_chg_5", vol20.diff(5).shift(1))
        _add_col(f"{t.lower()}_vol20_over_vol60", (vol20 / (vol60 + 1e-12)).shift(1))

    # 8) Volume anomaly lag/changes for key assets with volume information.
    for t in lead_lag_tickers:
        z_col = f"{t}_volume_z_20"
        if z_col not in df.columns:
            continue
        vz = df[z_col].astype(float)
        _add_col(f"{t.lower()}_volume_z_20_lag1", vz.shift(1))
        _add_col(f"{t.lower()}_volume_z_20_lag3", vz.shift(3))
        _add_col(f"{t.lower()}_volume_z_20_chg_1", vz.diff(1).shift(1))
        _add_col(f"{t.lower()}_volume_z_20_chg_5", vz.diff(5).shift(1))

    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1).copy()

    return df, added_cols


def expand_with_other_stock_features(
    base_features_df: pd.DataFrame,
    data_dir: str | Path,
    exclude_stocks: set[str] | None = None,
    add_stock_technical_features: bool = True,
    add_cross_asset_features: bool = True,
    drop_high_nan_features: bool = False,
    nan_drop_threshold: float = 0.40,
    other_join: str = "inner",
    final_join: str = "left",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Expand a base feature table with prefixed features from other stocks.

    Parameters
    ----------
    base_features_df:
        The existing feature table to expand (e.g., AAPL_data_expand), indexed by datetime.
    data_dir:
        Directory containing stock CSV files with a 'Dt' column.
    exclude_stocks:
        Tickers to skip when building the other-stock feature table.
    add_stock_technical_features:
        If True, run add_technical_features for each other stock before prefixing.
    add_cross_asset_features:
        If True, append cross-asset relational features (breadth/dispersion/relative strength/
        spread momentum/rolling correlations) after joining other stock features.
    drop_high_nan_features:
        If True, drop columns with NaN ratio strictly greater than ``nan_drop_threshold``.
    nan_drop_threshold:
        Threshold for column-wise NaN ratio dropping, used only when
        ``drop_high_nan_features=True``.
    other_join:
        Join mode across other-stock feature blocks. Typically 'inner' for strict date alignment.
    final_join:
        Join mode when attaching the other-stock table onto base_features_df. Typically 'left'.

    Returns
    -------
    expanded_df, other_features_big_table, audit_info
    """

    if exclude_stocks is None:
        exclude_stocks = set()

    base = base_features_df.copy()
    if "Dt" in base.columns:
        base["Dt"] = pd.to_datetime(base["Dt"], format="%Y-%m-%d", errors="coerce")
        base = base.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")
    else:
        base = base.sort_index().copy()
        base.index = pd.to_datetime(base.index, errors="coerce")
        base = base[base.index.notna()]

    paths = sorted(Path(data_dir).glob("*.csv"))
    use_paths = [p for p in paths if p.stem not in set(exclude_stocks)]

    other_blocks = []
    used_tickers = []

    for path in use_paths:
        ticker = path.stem
        df = pd.read_csv(path)
        if "Dt" not in df.columns:
            continue

        df["Dt"] = pd.to_datetime(df["Dt"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")

        feat_df = add_technical_features(df) if add_stock_technical_features else df
        feat_df = feat_df.rename(columns={col: f"{ticker}_{col}" for col in feat_df.columns})

        other_blocks.append(feat_df)
        used_tickers.append(ticker)

    if not other_blocks:
        raise ValueError("No other stock feature blocks were created. Check data_dir/exclude_stocks.")

    other_features_big_table = pd.concat(other_blocks, axis=1, join=other_join).sort_index()
    expanded_df = base.join(other_features_big_table, how=final_join)

    cross_asset_added_cols: list[str] = []
    if add_cross_asset_features:
        expanded_df, cross_asset_added_cols = add_cross_asset_relation_features(expanded_df)

    dropped_high_nan_cols: list[str] = []
    if drop_high_nan_features:
        nan_ratio_all = expanded_df.isna().mean()
        dropped_high_nan_cols = nan_ratio_all[nan_ratio_all > nan_drop_threshold].index.tolist()
        if dropped_high_nan_cols:
            expanded_df = expanded_df.drop(columns=dropped_high_nan_cols)

    duplicated_cols = expanded_df.columns[expanded_df.columns.duplicated()].tolist()
    nan_ratio_top20 = expanded_df.isna().mean().sort_values(ascending=False).head(20)

    audit_info = {
        "used_tickers": used_tickers,
        "other_features_shape": other_features_big_table.shape,
        "base_shape": base.shape,
        "expanded_shape": expanded_df.shape,
        "other_date_min": other_features_big_table.index.min(),
        "other_date_max": other_features_big_table.index.max(),
        "duplicated_columns": duplicated_cols,
        "cross_asset_added_columns": cross_asset_added_cols,
        "dropped_high_nan_columns": dropped_high_nan_cols,
        "top_nan_ratio": nan_ratio_top20,
    }

    return expanded_df, other_features_big_table, audit_info


def build_aapl_expanded_features(
    aapl_data: pd.DataFrame,
    data_dir: str | Path,
    exclude_stocks: set[str] | None = None,
    drop_high_nan_features: bool = True,
    nan_drop_threshold: float = 0.40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the current AAPL expanded feature table used in the notebook.

    This collects the scattered notebook steps:
    1) AAPL technical features
    2) SPY rolling alpha/beta, SPY technical features, and AAPL excess returns
    3) final market-feature pass
    4) other-stock features plus cross-asset relation features

    Interaction features are intentionally not included here.
    """
    data_dir = Path(data_dir)
    spy_path = data_dir / "SPY.csv"

    if exclude_stocks is None:
        exclude_stocks = {"AAPL", "MA", "V", "FB", "XLRE"}

    aapl_expand = add_technical_features(aapl_data)

    aapl_expand = add_rolling_alpha_beta_from_ret1d(
        stock_features_df=aapl_expand,
        market_data=spy_path,
        rolling_window=30,
    )
    aapl_expand = add_rolling_alpha_beta_from_ret1d(
        stock_features_df=aapl_expand,
        market_data=spy_path,
        rolling_window=60,
    )

    if {"alpha_60", "alpha_30"}.issubset(aapl_expand.columns):
        aapl_expand["alpha_60minus_30"] = aapl_expand["alpha_60"] - aapl_expand["alpha_30"]
    if {"beta_60", "beta_30"}.issubset(aapl_expand.columns):
        aapl_expand["beta_60minus_30"] = aapl_expand["beta_60"] - aapl_expand["beta_30"]

    spy_raw = pd.read_csv(spy_path)
    if "Dt" in spy_raw.columns:
        spy_raw["Dt"] = pd.to_datetime(spy_raw["Dt"], format="%Y-%m-%d", errors="coerce")
        spy_raw = spy_raw.dropna(subset=["Dt"]).sort_values("Dt").drop_duplicates(subset="Dt").set_index("Dt")

    spy_features = build_market_features(spy_raw)
    aapl_expand = aapl_expand.join(spy_features, how="left")

    if {"ret_1d", "spy_ret_1d"}.issubset(aapl_expand.columns):
        aapl_expand["aapl_excess_ret_1d"] = aapl_expand["ret_1d"] - aapl_expand["spy_ret_1d"]
    if {"ret_ma_5", "spy_ret_ma_5"}.issubset(aapl_expand.columns):
        aapl_expand["aapl_excess_ret_ma_5"] = aapl_expand["ret_ma_5"] - aapl_expand["spy_ret_ma_5"]
    if {"ret_ma_20", "spy_ret_ma_20"}.issubset(aapl_expand.columns):
        aapl_expand["aapl_excess_ret_ma_20"] = aapl_expand["ret_ma_20"] - aapl_expand["spy_ret_ma_20"]

    aapl_expand = add_more_market_feature(aapl_expand, spy_path)

    aapl_expand, other_features_big_table, other_feature_audit = expand_with_other_stock_features(
        base_features_df=aapl_expand,
        data_dir=data_dir,
        exclude_stocks=exclude_stocks,
        add_stock_technical_features=True,
        add_cross_asset_features=True,
        drop_high_nan_features=drop_high_nan_features,
        nan_drop_threshold=nan_drop_threshold,
        other_join="inner",
        final_join="left",
    )

    return aapl_expand, other_features_big_table, other_feature_audit



def window_time_split(
    data: pd.DataFrame,
    target: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    window_size: int = 252,
    target_mode: str = "raw",
    return_index: bool = False,
    return_raw_target: bool = False,
):
    """Align data/target, then build rolling windows and split in time order."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    target_mode = target_mode.lower().strip()
    three_class_modes = {"direction_3class", "direction3", "categorical"}
    binary_modes = {"direction_binary", "direction_2class", "direction2", "binary"}
    zero_binary_modes = {"direction_binary_zero", "direction_2class_zero", "binary_zero"}
    filtered_binary_modes = {"direction_binary_filtered", "direction_2class_filtered", "binary_filtered"}
    if target_mode not in {"raw"} | three_class_modes | binary_modes | zero_binary_modes | filtered_binary_modes:
        raise ValueError(
            "target_mode must be one of {'raw', 'direction_3class', 'direction3', "
            "'categorical', 'direction_binary', 'direction_2class', 'direction2', 'binary', "
            "'direction_binary_zero', 'direction_2class_zero', 'binary_zero', "
            "'direction_binary_filtered', 'direction_2class_filtered', 'binary_filtered'}"
        )

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

    raw_target = target.copy()
    train_filtered_binary = target_mode in binary_modes
    drop_all_binary = target_mode in filtered_binary_modes
    binary_target = target_mode in binary_modes | zero_binary_modes | filtered_binary_modes
    binary_filter_target = None
    if target_mode in three_class_modes:
        target = pd.Series(_direction_3class_target(target.to_numpy()), index=target.index, name=target.name)
    elif target_mode in binary_modes | zero_binary_modes:
        target = pd.Series(_direction_binary_target(target.to_numpy()), index=target.index, name=target.name)
        if train_filtered_binary:
            binary_filter_target = pd.Series(
                _direction_binary_filtered_target(raw_target.to_numpy()),
                index=target.index,
                name=target.name,
            )
    elif target_mode in filtered_binary_modes:
        target = pd.Series(_direction_binary_filtered_target(target.to_numpy()), index=target.index, name=target.name)

    X_raw = data.to_numpy()
    y_raw = target.to_numpy()
    raw_y_raw = raw_target.to_numpy()
    filter_y_raw = binary_filter_target.to_numpy() if binary_filter_target is not None else None
    idx_raw = data.index.to_numpy()

    n = len(data)
    if n <= window_size:
        raise ValueError(f"Not enough rows ({n}) for window_size={window_size}")

    # 1) Window first
    X_list, y_list, raw_y_list, filter_y_list, t_list = [], [], [], [], []
    for t in range(window_size, n):
        X_list.append(X_raw[t - window_size:t])  # past window
        y_list.append(y_raw[t])                  # predict current time t
        raw_y_list.append(raw_y_raw[t])          # original target value before label conversion
        if filter_y_raw is not None:
            filter_y_list.append(filter_y_raw[t])
        t_list.append(idx_raw[t])                # timestamp of label

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    raw_y = np.asarray(raw_y_list)
    filter_y = np.asarray(filter_y_list) if filter_y_raw is not None else None
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
    raw_y_train, raw_y_val = raw_y[:train_end], raw_y[train_end:val_end]
    raw_y_test = raw_y[val_end:]
    if filter_y is not None:
        filter_y_train = filter_y[:train_end]
        filter_y_val = filter_y[train_end:val_end]
        filter_y_test = filter_y[val_end:]

    if return_index:
        idx_train = t_idx[:train_end]
        idx_val = t_idx[train_end:val_end]
        idx_test = t_idx[val_end:]

        if train_filtered_binary or drop_all_binary:
            if train_filtered_binary:
                train_mask = filter_y_train != -1
                val_mask = np.ones(len(y_val), dtype=bool)
                test_mask = np.ones(len(y_test), dtype=bool)
            else:
                train_mask = y_train != -1
                val_mask = y_val != -1
                test_mask = y_test != -1
            x_train, y_train, idx_train = x_train[train_mask], y_train[train_mask].astype(int), idx_train[train_mask]
            x_val, y_val, idx_val = x_val[val_mask], y_val[val_mask].astype(int), idx_val[val_mask]
            x_test, y_test, idx_test = x_test[test_mask], y_test[test_mask].astype(int), idx_test[test_mask]
            raw_y_train, raw_y_val, raw_y_test = raw_y_train[train_mask], raw_y_val[val_mask], raw_y_test[test_mask]
            if min(len(y_train), len(y_val), len(y_test)) == 0:
                raise ValueError(f"{target_mode} produced an empty train, validation, or test split")
        elif binary_target:
            y_train, y_val, y_test = y_train.astype(int), y_val.astype(int), y_test.astype(int)

        if return_raw_target:
            return (
                x_train, y_train, x_val, y_val, x_test, y_test,
                idx_train, idx_val, idx_test,
                raw_y_train, raw_y_val, raw_y_test,
            )

        return x_train, y_train, x_val, y_val, x_test, y_test, idx_train, idx_val, idx_test

    if train_filtered_binary or drop_all_binary:
        if train_filtered_binary:
            train_mask = filter_y_train != -1
            val_mask = np.ones(len(y_val), dtype=bool)
            test_mask = np.ones(len(y_test), dtype=bool)
        else:
            train_mask = y_train != -1
            val_mask = y_val != -1
            test_mask = y_test != -1
        x_train, y_train = x_train[train_mask], y_train[train_mask].astype(int)
        x_val, y_val = x_val[val_mask], y_val[val_mask].astype(int)
        x_test, y_test = x_test[test_mask], y_test[test_mask].astype(int)
        raw_y_train, raw_y_val, raw_y_test = raw_y_train[train_mask], raw_y_val[val_mask], raw_y_test[test_mask]
        if min(len(y_train), len(y_val), len(y_test)) == 0:
            raise ValueError(f"{target_mode} produced an empty train, validation, or test split")
    elif binary_target:
        y_train, y_val, y_test = y_train.astype(int), y_val.astype(int), y_test.astype(int)

    if return_raw_target:
        return x_train, y_train, x_val, y_val, x_test, y_test, raw_y_train, raw_y_val, raw_y_test

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
    return_raw_target: bool = False,
):
    """
    Build sliding windows first, then split by time order.
    X_df: feature dataframe, indexed by time
    y_series: target series, indexed by time

    add_transformed_features:
    - 0/False: keep original features only
    - 1/True: add transformed features and keep all original columns
    - 2: keep only transformed features
    """
    data = _apply_transformed_feature_mode(X_df.sort_index().copy(), add_transformed_features)

    return window_time_split(
        data=data,
        target=y_series,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        window_size=window_size,
        target_mode=target_mode,
        return_index=return_index,
        return_raw_target=return_raw_target,
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
    return_raw_target: bool = False,
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
        - 0/False: keep original main-stock features only
        - 1/True: add transformed main-stock features and keep all original columns
        - 2: keep only transformed main-stock features
    """
    data = _apply_transformed_feature_mode(X_df.sort_index().copy(), add_transformed_features)

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
        return_raw_target=return_raw_target,
    )


def evaluate_keras_model_on_validation(
    model,
    history,
    x_val: np.ndarray,
    y_val_cat: np.ndarray,
    prefix: str = "model",
    class_names: tuple[str, ...] | None = None,
    raw_return: np.ndarray | pd.Series | None = None,
    benchmark_return: np.ndarray | pd.Series | None = None,
    benchmark_name: str | None = None,
    down_position: float = -1.0,
    flat_position: float = 0.0,
    up_position: float = 1.0,
    plot_history: bool = True,
):
    """Evaluate a trained Keras classification model on a validation set."""
    import importlib

    tf = importlib.import_module("tensorflow")
    keras = importlib.import_module("keras")
    metrics = importlib.import_module("sklearn.metrics")
    plt = importlib.import_module("matplotlib.pyplot")

    num_classes = int(model.output_shape[-1])
    if class_names is None or len(class_names) != num_classes:
        class_names = ("Down", "Up") if num_classes == 2 else ("Down", "Flat", "Up")
    labels = list(range(num_classes))

    y_val_rnn = np.asarray(y_val_cat)
    unique_values = set(np.unique(y_val_rnn).tolist())
    sample_mask = None

    if num_classes == 2:
        if np.issubdtype(y_val_rnn.dtype, np.integer) and unique_values.issubset({0, 1}):
            y_val_rnn = y_val_rnn.astype(int)
        else:
            y_val_rnn = _direction_binary_target(y_val_rnn)
            keep_mask = y_val_rnn != -1
            sample_mask = keep_mask
            x_val = x_val[keep_mask]
            y_val_rnn = y_val_rnn[keep_mask].astype(int)
    else:
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
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
    ))
    print(f"\n[{prefix}] Validation confusion matrix:")
    print(metrics.confusion_matrix(y_val_rnn, y_val_pred, labels=labels))

    trading_results = {}
    if raw_return is not None:
        raw_return_arr = np.asarray(raw_return, dtype=float).reshape(-1)
        if sample_mask is not None and len(raw_return_arr) == len(sample_mask):
            raw_return_arr = raw_return_arr[sample_mask]
        if len(raw_return_arr) != len(y_val_pred):
            raise ValueError(
                f"raw_return length ({len(raw_return_arr)}) must match validation predictions ({len(y_val_pred)})."
            )

        if num_classes == 2:
            position = np.where(y_val_pred == 1, up_position, down_position)
        elif num_classes == 3:
            position = np.where(
                y_val_pred == 2,
                up_position,
                np.where(y_val_pred == 0, down_position, flat_position),
            )
        else:
            raise ValueError("Trading evaluation supports only 2-class or 3-class outputs.")

        strategy_return = position * raw_return_arr
        if benchmark_return is None:
            benchmark_arr = raw_return_arr
            benchmark_label = benchmark_name or "buy-and-hold"
        else:
            benchmark_arr = np.asarray(benchmark_return, dtype=float).reshape(-1)
            if sample_mask is not None and len(benchmark_arr) == len(sample_mask):
                benchmark_arr = benchmark_arr[sample_mask]
            if len(benchmark_arr) != len(y_val_pred):
                raise ValueError(
                    f"benchmark_return length ({len(benchmark_arr)}) must match validation predictions ({len(y_val_pred)})."
                )
            benchmark_label = benchmark_name or "benchmark"

        excess_return = strategy_return - benchmark_arr
        strategy_average_return = float(np.mean(strategy_return))
        benchmark_average_return = float(np.mean(benchmark_arr))
        excess_average_return = float(np.mean(excess_return))
        strategy_annualized_return = strategy_average_return * 252.0
        benchmark_annualized_return = benchmark_average_return * 252.0
        excess_annualized_return = excess_average_return * 252.0

        print(f"\n[{prefix}] Validation trading performance:")
        print(f"[{prefix}] Strategy average return: {strategy_average_return:.6%}")
        print(f"[{prefix}] {benchmark_label} average return: {benchmark_average_return:.6%}")
        print(f"[{prefix}] Excess average return: {excess_average_return:.6%}")
        print(f"[{prefix}] Strategy annualized arithmetic return: {strategy_annualized_return:.4%}")
        print(f"[{prefix}] {benchmark_label} annualized arithmetic return: {benchmark_annualized_return:.4%}")
        print(f"[{prefix}] Excess annualized arithmetic return: {excess_annualized_return:.4%}")

        trading_results = {
            "position": position,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_arr,
            "excess_return": excess_return,
            "strategy_average_return": strategy_average_return,
            "benchmark_average_return": benchmark_average_return,
            "excess_average_return": excess_average_return,
            "strategy_annualized_return": strategy_annualized_return,
            "benchmark_annualized_return": benchmark_annualized_return,
            "excess_annualized_return": excess_annualized_return,
            "mean_excess_return": excess_average_return,
        }

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

    results = {
        "val_loss": val_loss,
        "val_macro_f1": val_macro_f1,
        "y_val_pred": y_val_pred,
    }
    results.update(trading_results)
    return results
