from __future__ import annotations

import gc
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset_timeseries_cleaned_v2.csv"
OUT_DIR = ROOT / "lncs_figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024 if DEVICE.type == "cuda" else 512
OUTER_CAMPAIGNS = [4, 5, 6, 7]
SHAP_BACKGROUND_WINDOWS = 64
SHAP_WINDOWS_PER_FLIGHT = 16
SHAP_BATCH_SIZE = 96
BOOTSTRAP_REPS = 2000
BEST = {
    "window": 64,
    "hidden_size": 64,
    "layers": 2,
    "dropout": 0.05,
    "epochs": 55,
    "learning_rate": 7e-4,
    "weight_decay": 1e-4,
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def campaign_from_flight(f: int) -> int:
    if f <= 28:
        return 1
    if f <= 38:
        return 2
    if f <= 53:
        return 3
    if f <= 71:
        return 4
    if f <= 89:
        return 5
    if f <= 116:
        return 6
    return 7


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    data = pd.read_csv(DATA_PATH)
    data["flight_num"] = data["flight_id"].str.extract(r"(\d+)").astype(int)
    data = data.sort_values(["flight_num", "t_sec"]).reset_index(drop=True)
    data["campaign_id"] = data["flight_num"].map(campaign_from_flight)
    data["elapsed_s"] = data.groupby("flight_num", sort=False)["t_sec"].transform(lambda s: s - s.iloc[0]).clip(lower=0)
    data["log_elapsed_s"] = np.log1p(data["elapsed_s"])
    for period in [10, 30, 60, 120]:
        data[f"time_sin_{period}s"] = np.sin(2 * np.pi * data["elapsed_s"] / period)
        data[f"time_cos_{period}s"] = np.cos(2 * np.pi * data["elapsed_s"] / period)
    data["ground_speed_3d"] = np.sqrt(data["pos_vx"] ** 2 + data["pos_vy"] ** 2 + data["pos_vz"] ** 2)
    data["accel_magnitude"] = np.sqrt(data["pos_ax"] ** 2 + data["pos_ay"] ** 2 + data["pos_az"] ** 2)
    data["dynamic_pressure"] = 0.5 * data["air_rho"] * data["true_airspeed"] ** 2
    data["airspeed_cubed"] = data["true_airspeed"] ** 3
    data["climb_positive"] = data["climb_rate"].clip(lower=0)
    data["sink_positive"] = (-data["climb_rate"]).clip(lower=0)
    data["headwind_x_airspeed"] = data["headwind_component"] * data["true_airspeed"]
    data["load_factor_proxy"] = 1 / np.cos(data["tilt_angle"].clip(-1.2, 1.2))
    for angle in ["roll", "pitch"]:
        data[f"sin_{angle}"] = np.sin(data[angle])
        data[f"cos_{angle}"] = np.cos(data[angle])

    time_state = [
        "t_sec", "elapsed_s", "log_elapsed_s",
        "time_sin_10s", "time_cos_10s", "time_sin_30s", "time_cos_30s",
        "time_sin_60s", "time_cos_60s", "time_sin_120s", "time_cos_120s",
    ]
    base_state = [
        "pos_vx", "pos_vy", "pos_vz", "pos_ax", "pos_ay", "pos_az",
        "speed_horizontal", "climb_rate", "roll", "pitch", "tilt_angle",
        "headwind_component", "air_rho", "air_ambient_temperature",
        "true_airspeed", "wind_north", "wind_east",
        "ground_speed_3d", "accel_magnitude", "dynamic_pressure", "airspeed_cubed",
        "climb_positive", "sink_positive", "headwind_x_airspeed", "load_factor_proxy",
        "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
    ] + time_state
    history_sources = [
        "pos_vx", "pos_vy", "pos_vz", "pos_ax", "pos_ay", "pos_az",
        "speed_horizontal", "climb_rate", "roll", "pitch", "tilt_angle",
        "headwind_component", "true_airspeed", "t_sec", "log_elapsed_s",
    ]
    history_features = []
    for feature in history_sources:
        grouped = data.groupby("flight_num", sort=False)[feature]
        for lag in [1, 2, 3, 5, 10, 20]:
            name = f"{feature}_lag{lag}"
            data[name] = grouped.shift(lag)
            history_features.append(name)
        shifted = grouped.shift(1)
        shifted_group = shifted.groupby(data["flight_num"], sort=False)
        for window in [3, 5, 10, 20]:
            rolling = shifted_group.rolling(window, min_periods=1)
            for stat, values in [("mean", rolling.mean()), ("std", rolling.std()), ("min", rolling.min()), ("max", rolling.max())]:
                name = f"{feature}_{stat}{window}"
                data[name] = values.reset_index(level=0, drop=True)
                history_features.append(name)
            name = f"{feature}_ewm{window}"
            data[name] = shifted_group.transform(lambda s: s.ewm(span=window, adjust=False, min_periods=1).mean())
            history_features.append(name)
    return data, base_state, base_state + history_features


def clip_predictions(pred):
    return np.clip(np.asarray(pred, dtype=float), 0, None)


def regression_metrics(y_true, y_pred):
    pred = clip_predictions(y_pred)
    return {
        "RMSE_W": mean_squared_error(y_true, pred) ** 0.5,
        "MAE_W": mean_absolute_error(y_true, pred),
        "RMSLE": mean_squared_log_error(y_true, pred) ** 0.5,
        "R2": r2_score(y_true, pred),
    }


def extended_metrics(y_true, y_pred, groups):
    result = regression_metrics(y_true, y_pred)
    frame = pd.DataFrame({"y": np.asarray(y_true), "pred": clip_predictions(y_pred), "flight": np.asarray(groups)})
    dy, dp, per_flight = [], [], []
    for _, z in frame.groupby("flight", sort=False):
        per_flight.append(regression_metrics(z["y"], z["pred"]))
        dy.extend(np.diff(z["y"]))
        dp.extend(np.diff(z["pred"]))
    macro = pd.DataFrame(per_flight).mean()
    result.update({f"macro_{k}": v for k, v in macro.items()})
    result["delta_R2"] = r2_score(dy, dp)
    return result


def equal_flight_weights(groups):
    counts = pd.Series(groups).value_counts()
    w = pd.Series(groups).map(1 / counts).to_numpy(copy=True)
    return (w / w.mean()).astype(np.float32)


def build_sequence_windows(data, scaled_features, window):
    out = np.empty((len(scaled_features), window, scaled_features.shape[1]), dtype=np.float32)
    for _, flight in data.groupby("flight_num", sort=False):
        idx = flight.index.to_numpy()
        values = scaled_features[idx]
        padded = np.vstack([np.repeat(values[:1], window - 1, axis=0), values])
        view = np.lib.stride_tricks.sliding_window_view(padded, (window, values.shape[1]))[:, 0, :, :]
        out[idx] = view
    return out


class SequenceRNNRegressor(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1, :]).squeeze(-1)


class ShapOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


def make_loader(windows, indices, target_log, sample_weights, shuffle):
    ds = TensorDataset(
        torch.from_numpy(windows[indices]),
        torch.from_numpy(target_log[indices].astype(np.float32)),
        torch.from_numpy(sample_weights[indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=(DEVICE.type == "cuda"))


def train_best_gru(windows, target_log, train_indices, sample_weights, seed_offset):
    torch.manual_seed(SEED + seed_offset)
    model = SequenceRNNRegressor(windows.shape[-1], BEST["hidden_size"], BEST["layers"], BEST["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST["learning_rate"], weight_decay=BEST["weight_decay"])
    loader = make_loader(windows, train_indices, target_log, sample_weights, shuffle=True)
    for _ in range(BEST["epochs"]):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = ((pred - yb) ** 2 * wb).sum() / wb.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def predict_log_batches(model, x, batch_size=512):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).float().to(next(model.parameters()).device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds)


def compute_energy_tables(data, preds):
    all_actual = []
    for _, z in data.groupby("flight_num", sort=False):
        dt = float(z["t_sec"].diff().dropna().median())
        all_actual.append({
            "flight_num": int(z["flight_num"].iloc[0]),
            "campaign": int(z["campaign_id"].iloc[0]),
            "actual_Wh": float(z["power_w"].sum() * dt / 3600),
        })
    actual_energy = pd.DataFrame(all_actual)

    future_rows = []
    for (campaign, flight), z in preds.groupby(["campaign_id", "flight_num"], sort=False):
        dt = float(z["t_sec"].diff().dropna().median())
        actual = float(z["power_w"].sum() * dt / 3600)
        pred = float(z["prediction_w"].sum() * dt / 3600)
        future_rows.append({"campaign": int(campaign), "flight_num": int(flight), "actual_Wh": actual, "predicted_Wh": pred})
    flight_energy = pd.DataFrame(future_rows)
    threshold = float(actual_energy.loc[actual_energy["campaign"].le(3), "actual_Wh"].median())
    flight_energy["actual_watch"] = flight_energy["actual_Wh"] >= threshold
    flight_energy["predicted_watch"] = flight_energy["predicted_Wh"] >= threshold
    flight_energy["abs_percent_error"] = (flight_energy["actual_Wh"] - flight_energy["predicted_Wh"]).abs() / flight_energy["actual_Wh"].clip(lower=1e-9) * 100
    campaign_energy = flight_energy.groupby("campaign").agg(
        flights=("flight_num", "count"),
        actual_mean_Wh=("actual_Wh", "mean"),
        predicted_mean_Wh=("predicted_Wh", "mean"),
        MAE_Wh=("actual_Wh", lambda s: np.nan),
        MdAPE_percent=("abs_percent_error", "median"),
    ).reset_index()
    mae = flight_energy.groupby("campaign").apply(lambda z: (z["actual_Wh"] - z["predicted_Wh"]).abs().mean(), include_groups=False)
    campaign_energy["MAE_Wh"] = campaign_energy["campaign"].map(mae)
    n_top = max(1, int(np.ceil(0.2 * len(flight_energy))))
    top = flight_energy.sort_values("predicted_Wh", ascending=False).head(n_top)
    summary = pd.Series({
        "future_flights": float(len(flight_energy)),
        "training_policy_threshold_Wh": threshold,
        "median_actual_energy_Wh": float(flight_energy["actual_Wh"].median()),
        "median_predicted_energy_Wh": float(flight_energy["predicted_Wh"].median()),
        "mean_absolute_energy_error_Wh": float((flight_energy["actual_Wh"] - flight_energy["predicted_Wh"]).abs().mean()),
        "median_absolute_percent_error": float(flight_energy["abs_percent_error"].median()),
        "energy_rank_spearman": float(flight_energy[["actual_Wh", "predicted_Wh"]].corr(method="spearman").iloc[0, 1]),
        "top20_predicted_flights_actual_energy_share_percent": float(top["actual_Wh"].sum() / flight_energy["actual_Wh"].sum() * 100),
    })
    return flight_energy, campaign_energy, summary


def random_row_reference(data, tabular_features):
    random_train, random_test = train_test_split(np.arange(len(data)), test_size=0.20, random_state=SEED, shuffle=True)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(5, 95))),
        ("model", Ridge(alpha=30)),
    ])
    model.fit(data.iloc[random_train][tabular_features], np.log1p(data.iloc[random_train]["power_w"]))
    random_pred = clip_predictions(np.expm1(model.predict(data.iloc[random_test][tabular_features])))
    return regression_metrics(data.iloc[random_test]["power_w"], random_pred)


def write_rolling_figure(folds, aggregate):
    models = pd.read_csv(OUT_DIR / "model_reference_metrics.csv")[["model", "RMSLE"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.35))
    colors = ["#b84a4a", "#9f8f56", "#c98840", "#7d6bb3", "#6f5aa7", "#2f8f83"]
    axes[0].barh(models["model"], models["RMSLE"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Aggregate RMSLE")
    axes[0].set_title("Rolling future benchmark")
    for i, v in enumerate(models["RMSLE"]):
        axes[0].text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=8)
    axes[1].plot(folds["campaign"], folds["RMSLE"], marker="o", color="#2f8f83")
    for _, r in folds.iterrows():
        axes[1].text(r["campaign"], r["RMSLE"] + 0.004, f"{r['RMSLE']:.3f}", ha="center", fontsize=8)
    axes[1].set_title("GRU fold RMSLE")
    axes[1].set_xlabel("Future campaign")
    axes[1].set_ylabel("RMSLE")
    axes[2].plot(folds["campaign"], folds["R2"], marker="o", label="$R^2$", color="#2f6f9f")
    axes[2].plot(folds["campaign"], folds["delta_R2"], marker="o", label="$\\Delta R^2$", color="#c98840")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Level and local-change fit")
    axes[2].set_xlabel("Future campaign")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "rolling_performance_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_diagnostics_figure(preds, random_metrics, aggregate):
    diag = pd.DataFrame({"actual": preds["power_w"], "prediction": preds["prediction_w"], "flight": preds["flight_num"], "campaign": preds["campaign_id"], "t_sec": preds["t_sec"]})
    diag["residual"] = diag["actual"] - diag["prediction"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    axes[0, 0].bar(["Valid rolling\nfuture GRU", "Invalid random\nrow reference"], [aggregate["RMSLE"].iloc[0], random_metrics["RMSLE"]], color=["#2f8f83", "#b84a4a"])
    axes[0, 0].set_ylabel("RMSLE")
    axes[0, 0].set_title("Protocol audit")
    for i, v in enumerate([aggregate["RMSLE"].iloc[0], random_metrics["RMSLE"]]):
        axes[0, 0].text(i, v + 0.006, f"{v:.4f}", ha="center", fontsize=8)
    axes[0, 1].scatter(diag["actual"], diag["prediction"], s=6, alpha=0.18, color="#2f6f9f")
    lim = [0, max(diag["actual"].max(), diag["prediction"].max())]
    axes[0, 1].plot(lim, lim, "r--", linewidth=1)
    axes[0, 1].set_title("Actual vs predicted")
    axes[0, 1].set_xlabel("Actual W")
    axes[0, 1].set_ylabel("Predicted W")
    axes[0, 2].scatter(diag["prediction"], diag["residual"], s=6, alpha=0.18, color="#c98840")
    axes[0, 2].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 2].set_title("Residual structure")
    axes[0, 2].set_xlabel("Predicted W")
    axes[0, 2].set_ylabel("Residual W")
    sns.histplot(diag["residual"], bins=70, kde=True, ax=axes[1, 0], color="#6f5aa7")
    axes[1, 0].set_title("Residual distribution")
    stats.probplot(diag["residual"], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q diagnostic")
    for campaign, color in zip(OUTER_CAMPAIGNS, ["#2f8f83", "#b84a4a", "#2f6f9f", "#c98840"]):
        flight = diag.loc[diag["campaign"] == campaign, "flight"].iloc[0]
        z = diag[(diag["campaign"] == campaign) & (diag["flight"] == flight)].head(180)
        axes[1, 2].plot(z["t_sec"], z["actual"], color=color, linewidth=1.1, label=f"C{campaign} actual")
        axes[1, 2].plot(z["t_sec"], z["prediction"], color=color, linestyle="--", linewidth=1.0, label=f"C{campaign} pred")
    axes[1, 2].set_title("Representative future traces")
    axes[1, 2].set_xlabel("Seconds from trace start")
    axes[1, 2].set_ylabel("Power W")
    axes[1, 2].legend(fontsize=6)
    plt.suptitle("Final optimized GRU leakage audit and residual diagnostics", y=1.02, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "tsec_audit_diagnostics_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_shap(data, features):
    target_log = np.log1p(data["power_w"].to_numpy()).astype(np.float32)
    prior_mask = data["campaign_id"] < 7
    future_mask = data["campaign_id"].isin(OUTER_CAMPAIGNS)
    prior_indices = np.flatnonzero(prior_mask)
    scaler = StandardScaler().fit(data.loc[prior_mask, features])
    windows = build_sequence_windows(data, scaler.transform(data[features]).astype(np.float32), BEST["window"])
    weights = np.zeros(len(data), dtype=np.float32)
    weights[prior_indices] = equal_flight_weights(data.loc[prior_mask, "flight_num"])
    model = train_best_gru(windows, target_log, prior_indices, weights, seed_offset=7)
    model = ShapOutputWrapper(model.cpu().eval()).eval()
    rng = np.random.default_rng(SEED + 707)
    background_idx = rng.choice(prior_indices, size=SHAP_BACKGROUND_WINDOWS, replace=False)
    explain_idx = []
    for _, group in data.loc[future_mask].groupby("flight_num", sort=True):
        idx = group.index.to_numpy()
        if len(idx) <= SHAP_WINDOWS_PER_FLIGHT:
            chosen = idx
        else:
            positions = np.linspace(0, len(idx) - 1, SHAP_WINDOWS_PER_FLIGHT).round().astype(int)
            chosen = idx[np.unique(positions)]
        explain_idx.extend(chosen.tolist())
    explain_idx = np.asarray(explain_idx, dtype=int)
    explain_windows = windows[explain_idx]
    background = torch.from_numpy(windows[background_idx]).float()
    explainer = shap.GradientExplainer(model, background)
    value_chunks = []
    for start in range(0, len(explain_idx), SHAP_BATCH_SIZE):
        explain = torch.from_numpy(explain_windows[start:start + SHAP_BATCH_SIZE]).float()
        values = explainer.shap_values(explain)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 4:
            values = values[..., 0]
        value_chunks.append(values)
    values = np.concatenate(value_chunks, axis=0)
    abs_values = np.abs(values)
    base_model = model.model.eval()
    pred_log = predict_log_batches(base_model, explain_windows, BATCH_SIZE)
    pred_w = np.expm1(pred_log)
    watt_values = values * np.exp(pred_log)[:, None, None]
    abs_watt_values = np.abs(watt_values)
    meta = data.loc[explain_idx, ["campaign_id", "flight_num"]].reset_index(drop=True)

    per_sample_feature = abs_watt_values.mean(axis=1)
    per_sample_log_feature = abs_values.mean(axis=1)
    sample_long = pd.DataFrame(per_sample_feature, columns=features)
    sample_log_long = pd.DataFrame(per_sample_log_feature, columns=features)
    sample_long["campaign"] = meta["campaign_id"].to_numpy()
    sample_long["flight_num"] = meta["flight_num"].to_numpy()
    sample_log_long["campaign"] = meta["campaign_id"].to_numpy()
    sample_log_long["flight_num"] = meta["flight_num"].to_numpy()
    flight_feature = sample_long.groupby(["campaign", "flight_num"], sort=True)[features].mean().reset_index()
    flight_log_feature = sample_log_long.groupby(["campaign", "flight_num"], sort=True)[features].mean().reset_index()
    feature_by_flight = flight_feature[features].to_numpy()
    log_feature_by_flight = flight_log_feature[features].to_numpy()
    boot = np.empty((BOOTSTRAP_REPS, len(features)), dtype=np.float32)
    boot_log = np.empty((BOOTSTRAP_REPS, len(features)), dtype=np.float32)
    n_flights = len(flight_feature)
    for b in range(BOOTSTRAP_REPS):
        sample = rng.integers(0, n_flights, size=n_flights)
        boot[b] = feature_by_flight[sample].mean(axis=0)
        boot_log[b] = log_feature_by_flight[sample].mean(axis=0)
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    log_ci_low = np.percentile(boot_log, 2.5, axis=0)
    log_ci_high = np.percentile(boot_log, 97.5, axis=0)

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": feature_by_flight.mean(axis=0),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_abs_log_shap": log_feature_by_flight.mean(axis=0),
        "log_ci_low": log_ci_low,
        "log_ci_high": log_ci_high,
        "recent_abs_shap": abs_watt_values[:, -10:, :].mean(axis=(0, 1)),
        "older_abs_shap": abs_watt_values[:, :-10, :].mean(axis=(0, 1)),
    }).sort_values("mean_abs_shap", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)

    y_true = data.loc[explain_idx, "power_w"].to_numpy()
    baseline_pred = clip_predictions(pred_w)
    baseline_metrics = regression_metrics(y_true, baseline_pred)
    permutation_rows = []
    for j, feature in enumerate(features):
        permuted = explain_windows.copy()
        order = rng.permutation(len(permuted))
        permuted[:, :, j] = permuted[order, :, j]
        perm_log = predict_log_batches(base_model, permuted, BATCH_SIZE)
        perm_pred = np.expm1(perm_log)
        metrics = regression_metrics(y_true, perm_pred)
        permutation_rows.append({
            "feature": feature,
            "baseline_RMSE_W": baseline_metrics["RMSE_W"],
            "permuted_RMSE_W": metrics["RMSE_W"],
            "delta_RMSE_W": metrics["RMSE_W"] - baseline_metrics["RMSE_W"],
            "baseline_MAE_W": baseline_metrics["MAE_W"],
            "permuted_MAE_W": metrics["MAE_W"],
            "delta_MAE_W": metrics["MAE_W"] - baseline_metrics["MAE_W"],
            "baseline_RMSLE": baseline_metrics["RMSLE"],
            "permuted_RMSLE": metrics["RMSLE"],
            "delta_RMSLE": metrics["RMSLE"] - baseline_metrics["RMSLE"],
        })
    permutation = pd.DataFrame(permutation_rows).sort_values("delta_RMSE_W", ascending=False)
    permutation["rank"] = np.arange(1, len(permutation) + 1)

    campaign_rows = []
    for campaign, z in flight_feature.groupby("campaign", sort=True):
        means = z[features].mean().sort_values(ascending=False)
        for rank, (feature, value) in enumerate(means.items(), start=1):
            campaign_rows.append({
                "campaign": int(campaign),
                "feature": feature,
                "mean_abs_shap": float(value),
                "rank": rank,
            })
    campaign_importance = pd.DataFrame(campaign_rows)
    stability = []
    for campaign, z in campaign_importance.groupby("campaign", sort=True):
        top = z.sort_values("rank").iloc[0]
        sin_pitch = z.loc[z["feature"].eq("sin_pitch")].iloc[0]
        stability.append({
            "campaign": int(campaign),
            "top_feature": top["feature"],
            "top_mean_abs_shap": float(top["mean_abs_shap"]),
            "sin_pitch_rank": int(sin_pitch["rank"]),
            "sin_pitch_mean_abs_shap": float(sin_pitch["mean_abs_shap"]),
            "sin_pitch_is_top": bool(sin_pitch["rank"] == 1),
        })
    stability = pd.DataFrame(stability)
    temporal = pd.DataFrame({
        "window_position": np.arange(BEST["window"]),
        "seconds_before_prediction": (BEST["window"] - 1 - np.arange(BEST["window"])) * 0.506,
        "mean_abs_shap": abs_watt_values.mean(axis=(0, 2)),
        "mean_abs_log_shap": abs_values.mean(axis=(0, 2)),
    })
    recent_mass = float(abs_watt_values[:, -10:, :].sum() / abs_watt_values.sum() * 100)
    summary = pd.Series({
        "background_windows": float(SHAP_BACKGROUND_WINDOWS),
        "explained_windows": float(len(explain_idx)),
        "explained_future_flights": float(n_flights),
        "windows_per_future_flight": float(SHAP_WINDOWS_PER_FLIGHT),
        "recent_0_5s_attribution_mass_percent": recent_mass,
        "sin_pitch_top_in_all_campaigns": bool(stability["sin_pitch_is_top"].all()),
        "sin_pitch_worst_campaign_rank": float(stability["sin_pitch_rank"].max()),
        "permutation_baseline_RMSE_W": baseline_metrics["RMSE_W"],
        "permutation_baseline_MAE_W": baseline_metrics["MAE_W"],
    })
    importance.to_csv(OUT_DIR / "gru_final_shap_importance.csv", index=False)
    flight_feature.to_csv(OUT_DIR / "gru_final_shap_by_flight.csv", index=False)
    campaign_importance.to_csv(OUT_DIR / "gru_final_shap_by_campaign.csv", index=False)
    stability.to_csv(OUT_DIR / "gru_final_shap_stability.csv", index=False)
    permutation.to_csv(OUT_DIR / "gru_final_permutation_importance.csv", index=False)
    temporal.to_csv(OUT_DIR / "gru_final_shap_temporal.csv", index=False)
    summary.to_csv(OUT_DIR / "gru_final_shap_temporal_summary.csv")
    del model, windows
    gc.collect()
    return importance, summary


def compute_c7_affine_adaptation(preds):
    rows = []
    c7 = preds.loc[preds["campaign_id"].eq(7)].copy()
    c7_flights = sorted(c7["flight_num"].unique())
    for n_adapt in [1, 2]:
        adapt_flights = c7_flights[:n_adapt]
        eval_flights = c7_flights[n_adapt:]
        adapt = c7.loc[c7["flight_num"].isin(adapt_flights)]
        eval_set = c7.loc[c7["flight_num"].isin(eval_flights)]
        x_adapt = np.log1p(clip_predictions(adapt["prediction_w"].to_numpy()))
        y_adapt = np.log1p(adapt["power_w"].to_numpy())
        design = np.column_stack([x_adapt, np.ones_like(x_adapt)])
        slope, intercept = np.linalg.lstsq(design, y_adapt, rcond=None)[0]
        baseline = regression_metrics(eval_set["power_w"], eval_set["prediction_w"])
        x_eval = np.log1p(clip_predictions(eval_set["prediction_w"].to_numpy()))
        adapted_pred = np.expm1(slope * x_eval + intercept)
        adapted = regression_metrics(eval_set["power_w"], adapted_pred)
        for setting, metrics in [("baseline_no_adaptation", baseline), ("affine_log_calibration", adapted)]:
            rows.append({
                "adaptation_flights": n_adapt,
                "adaptation_flight_ids": ",".join(map(str, adapt_flights)),
                "evaluation_flight_ids": ",".join(map(str, eval_flights)),
                "setting": setting,
                "slope": float(slope) if setting == "affine_log_calibration" else 1.0,
                "intercept": float(intercept) if setting == "affine_log_calibration" else 0.0,
                **{k: float(v) for k, v in metrics.items()},
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "c7_affine_adaptation.csv", index=False)
    return result


def compute_c7_trajectory_diagnostics(data, preds):
    pred_metrics = []
    for (campaign, flight), z in preds.groupby(["campaign_id", "flight_num"], sort=True):
        residual = z["power_w"] - z["prediction_w"]
        pred_metrics.append({
            "campaign": int(campaign),
            "flight_num": int(flight),
            "mean_actual_power_W": float(z["power_w"].mean()),
            "mean_predicted_power_W": float(z["prediction_w"].mean()),
            "mean_residual_W": float(residual.mean()),
            "rmse_W": float(mean_squared_error(z["power_w"], z["prediction_w"]) ** 0.5),
        })
    pred_metrics = pd.DataFrame(pred_metrics)

    rows = []
    for flight, z in data.groupby("flight_num", sort=True):
        z = z.sort_values("t_sec")
        dx = z["pos_x"].diff().fillna(0)
        dy = z["pos_y"].diff().fillna(0)
        dz = z["pos_z"].diff().fillna(0)
        horizontal_step = np.sqrt(dx ** 2 + dy ** 2)
        path_step = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        rows.append({
            "campaign": int(z["campaign_id"].iloc[0]),
            "flight_num": int(flight),
            "duration_s": float(z["t_sec"].iloc[-1] - z["t_sec"].iloc[0]),
            "horizontal_distance_m": float(horizontal_step.sum()),
            "path_length_m": float(path_step.sum()),
            "altitude_range_m": float(z["air_baro_alt_meter"].max() - z["air_baro_alt_meter"].min()),
            "vertical_displacement_m": float(z["air_baro_alt_meter"].iloc[-1] - z["air_baro_alt_meter"].iloc[0]),
            "mean_abs_climb_rate": float(z["climb_rate"].abs().mean()),
            "climb_fraction": float((z["climb_rate"] > 0.2).mean()),
            "sink_fraction": float((z["climb_rate"] < -0.2).mean()),
            "mean_true_airspeed": float(z["true_airspeed"].mean()),
            "mean_speed_horizontal": float(z["speed_horizontal"].mean()),
            "mean_headwind_component": float(z["headwind_component"].mean()),
            "mean_wind_speed": float(np.sqrt(z["wind_north"] ** 2 + z["wind_east"] ** 2).mean()),
            "mean_air_density": float(z["air_rho"].mean()),
            "mean_air_temperature": float(z["air_ambient_temperature"].mean()),
            "mean_pitch": float(z["pitch"].mean()),
            "mean_tilt_angle": float(z["tilt_angle"].mean()),
            "mean_dynamic_pressure": float((0.5 * z["air_rho"] * z["true_airspeed"] ** 2).mean()),
        })
    flight = pd.DataFrame(rows).merge(pred_metrics, on=["campaign", "flight_num"], how="left")
    pre = flight.loc[flight["campaign"].lt(7)]
    c7 = flight.loc[flight["campaign"].eq(7)]
    metrics = [c for c in flight.columns if c not in {"campaign", "flight_num"}]
    summary_rows = []
    for metric in metrics:
        pre_mean = float(pre[metric].mean())
        c7_mean = float(c7[metric].mean())
        pooled_sd = float(np.sqrt((pre[metric].var(ddof=1) + c7[metric].var(ddof=1)) / 2))
        smd = (c7_mean - pre_mean) / pooled_sd if pooled_sd > 0 else np.nan
        summary_rows.append({
            "metric": metric,
            "pre_C1_C6_mean": pre_mean,
            "C7_mean": c7_mean,
            "standardized_mean_difference": smd,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        "standardized_mean_difference",
        key=lambda s: s.abs(),
        ascending=False,
    )
    flight.to_csv(OUT_DIR / "c7_trajectory_flight_diagnostics.csv", index=False)
    summary.to_csv(OUT_DIR / "c7_trajectory_shift_summary.csv", index=False)
    return flight, summary


def write_dashboard(summary, campaign_energy, shap_importance, shap_summary, aggregate):
    reference = pd.read_csv(OUT_DIR / "model_reference_metrics.csv")
    colors = {
        "Linear pitch": "#b84a4a",
        "Physics proxy": "#9f8f56",
        "Ridge tabular": "#c98840",
        "LSTM state time": "#7d6bb3",
        "TCN state time": "#6f5aa7",
        "GRU state time": "#2f8f83",
    }
    labels = {
        "Linear pitch": "Linear",
        "Physics proxy": "Physics",
        "Ridge tabular": "Ridge",
        "LSTM state time": "LSTM",
        "TCN state time": "TCN",
        "GRU state time": "GRU",
    }
    max_rmsle = float(reference["RMSLE"].max())
    models = [(labels[r.model], float(r.RMSLE), colors[r.model]) for r in reference.itertuples()]
    bar_rows = []
    for i, (name, val, color) in enumerate(models):
        y = 92 + i * 43
        width = 210 * val / max_rmsle
        bar_rows.append(f'<rect x="122" y="{y}" width="{width:.1f}" height="25" fill="{color}"/><text x="20" y="{y+18}" font-size="14">{name}</text><text x="{122+width+6:.1f}" y="{y+18}" font-size="14">{val:.4f}</text>')
    # Hand-tuned SVG scaling preserves the existing dashboard layout.
    x_min, x_max = 8.0, 12.2
    y_min, y_max = 8.0, 10.8
    def sx(v): return 78 + (v - x_min) / (x_max - x_min) * (454 - 78)
    def sy(v): return 310 - (v - y_min) / (y_max - y_min) * (310 - 82)
    point_rows = []
    for _, r in campaign_energy.iterrows():
        x, y = sx(r["actual_mean_Wh"]), sy(r["predicted_mean_Wh"])
        point_rows.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="#d59557" stroke="#7b4a16" stroke-width="2"/><text x="{x+14:.1f}" y="{y-12:.1f}" font-size="17" font-weight="700">C{int(r["campaign"])}</text>')
    threshold = float(summary["training_policy_threshold_Wh"])
    thresh_x, thresh_y = sx(threshold), sy(threshold)
    top3 = shap_importance.head(3)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAV Energy Audit View</title><style>
:root{{--ink:#1f2933;--muted:#5f6f82;--line:#cfd8e3;--panel:#f7fafc;--blue:#2f6f9f;--teal:#2f8f83;--amber:#c98840;--red:#b84a4a;--violet:#6f5aa7;}}
*{{box-sizing:border-box}}body{{margin:0;width:900px;height:1120px;font-family:"Segoe UI",Arial,sans-serif;color:var(--ink);background:#fff}}
.page{{width:900px;height:1120px;padding:28px 34px;background:linear-gradient(90deg,rgba(207,216,227,.32) 1px,transparent 1px),linear-gradient(0deg,rgba(207,216,227,.32) 1px,transparent 1px);background-size:72px 72px}}
.title{{display:grid;grid-template-columns:1fr 250px;gap:18px;border-bottom:3px solid #243447;padding-bottom:14px;margin-bottom:16px;align-items:start}}
h1{{margin:0;font-size:35px;line-height:1.05;font-weight:760;letter-spacing:0}}.subtitle{{margin-top:8px;font-size:18px;line-height:1.32;color:var(--muted)}}.badge{{border:1px solid #98a6b5;background:rgba(255,255,255,.9);padding:12px 13px;font-size:15px;line-height:1.42}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}}.card{{background:#fff;border:1px solid #d4dde8;padding:12px 10px;min-height:100px}}.k{{color:var(--muted);font-size:14px}}.v{{font-size:29px;font-weight:760;margin-top:7px;line-height:1}}.u{{color:var(--muted);font-size:13px;margin-top:7px}}
.panel{{background:rgba(247,250,252,.95);border:1px solid var(--line);padding:16px;margin-bottom:14px}}h2{{margin:0 0 11px 0;font-size:21px;font-weight:740}}h3{{margin:0 0 8px 0;font-size:17px;font-weight:730}}.twocol{{display:grid;grid-template-columns:1.12fr .88fr;gap:14px;align-items:stretch}}.smallgrid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.box{{background:#fff;border:1px solid #d4dde8;padding:14px;min-height:165px}}ul{{margin:0;padding-left:20px;font-size:15.2px;line-height:1.48;color:#334155}}.note{{font-size:15.8px;line-height:1.45;color:#334155;background:#fff;border:1px solid #d4dde8;padding:14px}}.note b{{color:#172033}}svg{{display:block;width:100%;background:#fff;border:1px solid #d4dde8}}svg text{{font-family:"Segoe UI",Arial,sans-serif;fill:var(--ink)}}.axis{{stroke:#64748b;stroke-width:1.4}}.gridline{{stroke:#e2e8f0;stroke-width:1.1}}.watch{{stroke:var(--red);stroke-dasharray:6 5;stroke-width:2}}.footer{{margin-top:11px;color:var(--muted);font-size:14.3px;line-height:1.38}}
</style></head><body><main class="page">
<div class="title"><div><h1>UAV Energy Audit Support View</h1><div class="subtitle">Held-out C4-C7 predictions from the no-PWM GRU baseline. This is an energy review and model-audit view, not an operational dispatch tool.</div></div><div class="badge">Browser-rendered with Playwright<br>Source: executed final outputs<br>Usable pipeline: 120 flights -> 118 retained</div></div>
<div class="cards"><div class="card"><div class="k">Future flights</div><div class="v">{int(summary['future_flights'])}</div><div class="u">C4-C7 only</div></div><div class="card"><div class="k">Watch threshold</div><div class="v">{threshold:.3f}</div><div class="u">Wh from C1-C3</div></div><div class="card"><div class="k">Energy MdAPE</div><div class="v">{summary['median_absolute_percent_error']:.2f}%</div><div class="u">flight level</div></div><div class="card"><div class="k">Top quintile</div><div class="v">{summary['top20_predicted_flights_actual_energy_share_percent']:.2f}%</div><div class="u">actual energy share</div></div></div>
<section class="panel"><h2>Flight-Energy Evidence</h2><div class="twocol"><div>
<svg height="382" viewBox="0 0 500 382"><text x="24" y="34" font-size="19" font-weight="700">Energy Watch Map</text><text x="24" y="58" font-size="14" fill="#5f6f82">Campaign means; diagonal is perfect Wh agreement</text><line x1="78" y1="310" x2="454" y2="310" class="axis"/><line x1="78" y1="82" x2="78" y2="310" class="axis"/><line x1="78" y1="310" x2="454" y2="82" stroke="#9aa6b2" stroke-dasharray="5 5" stroke-width="1.6"/><line x1="78" y1="{thresh_y:.1f}" x2="454" y2="{thresh_y:.1f}" class="watch"/><line x1="{thresh_x:.1f}" y1="82" x2="{thresh_x:.1f}" y2="310" class="watch"/><text x="{thresh_x+8:.1f}" y="104" font-size="13" fill="#b84a4a">{threshold:.3f} Wh threshold</text>{''.join(point_rows)}<text x="152" y="356" font-size="15">Actual mean flight energy (Wh)</text><text x="24" y="240" font-size="15" transform="rotate(-90 24 240)">Predicted mean energy (Wh)</text></svg>
<div class="footer">C7 remains the visible difficult case: {campaign_energy.loc[campaign_energy['campaign'].eq(7),'actual_mean_Wh'].iloc[0]:.4f} actual vs {campaign_energy.loc[campaign_energy['campaign'].eq(7),'predicted_mean_Wh'].iloc[0]:.4f} predicted Wh.</div></div>
<div><svg height="382" viewBox="0 0 390 382"><text x="20" y="34" font-size="19" font-weight="700">Rolling Baseline</text><text x="20" y="58" font-size="14" fill="#5f6f82">RMSLE lower is better; no PWM inputs</text><line x1="122" y1="318" x2="350" y2="318" class="axis"/><line x1="122" y1="92" x2="122" y2="318" class="axis"/>{''.join(bar_rows)}<text x="20" y="350" font-size="14" fill="#5f6f82">Selected on dev folds; C4-C7 held out.</text></svg></div></div></section>
<section class="panel"><h2>Preliminary Attribution Evidence</h2><div class="smallgrid"><div class="box"><h3>GradientExplainer SHAP Audit</h3><ul><li>Selected model: GRU trained on C1-C6, explained on C4-C7.</li><li>Background windows: {int(shap_summary['background_windows'])}; explained windows: {int(shap_summary['explained_windows'])} across {int(shap_summary['explained_future_flights'])} future flights.</li><li>Top watt-space attributions: {top3.iloc[0]['feature']} {top3.iloc[0]['mean_abs_shap']:.2f} W, {top3.iloc[1]['feature']} {top3.iloc[1]['mean_abs_shap']:.2f} W, {top3.iloc[2]['feature']} {top3.iloc[2]['mean_abs_shap']:.2f} W.</li><li>sin_pitch top in every campaign: {bool(shap_summary['sin_pitch_top_in_all_campaigns'])}; worst rank {int(shap_summary['sin_pitch_worst_campaign_rank'])}.</li></ul></div><div class="box"><h3>Operational Meaning</h3><ul><li>Predictions become flight-level Wh indicators for charging and maintenance review.</li><li>The watch list is weak and only a human triage layer because top-quintile share is {summary['top20_predicted_flights_actual_energy_share_percent']:.2f}%.</li><li>Attribution supports audit transparency, not causal proof or automated command.</li></ul></div></div></section>
<section class="note"><h3>What This View Contributes</h3><b>Primary contribution:</b> a leakage-audited no-PWM rolling baseline for instantaneous fixed-wing UAV power estimation on the usable cleaned IDF DS subset.<br><br><b>Secondary evidence:</b> the same held-out predictions are translated into flight-energy summaries and preliminary GRU explanations, so reviewers can see both useful signal and failure modes.<br><br><b>Not claimed:</b> this is not a deployed dispatch system or hardware validation.</section>
</main></body></html>"""
    (OUT_DIR / "smartcity_management_dashboard.html").write_text(html, encoding="utf-8")


def main():
    seed_everything()
    OUT_DIR.mkdir(exist_ok=True)
    data, sequence_features, tabular_features = load_data()
    preds = pd.read_csv(OUT_DIR / "gru_final_outer_predictions.csv")
    folds = pd.read_csv(OUT_DIR / "gru_final_fold_metrics.csv")
    aggregate = pd.read_csv(OUT_DIR / "gru_final_aggregate_metrics.csv")
    flight_energy, campaign_energy, smart_summary = compute_energy_tables(data, preds)
    flight_energy.to_csv(OUT_DIR / "gru_final_flight_energy.csv", index=False)
    campaign_energy.to_csv(OUT_DIR / "gru_final_campaign_energy.csv", index=False)
    smart_summary.to_csv(OUT_DIR / "gru_final_smartcity_summary.csv")
    random_metrics = random_row_reference(data, tabular_features)
    shap_importance, shap_summary = compute_shap(data, sequence_features)
    c7_adaptation = compute_c7_affine_adaptation(preds)
    c7_flight_diag, c7_shift = compute_c7_trajectory_diagnostics(data, preds)
    write_rolling_figure(folds, aggregate)
    write_diagnostics_figure(preds, random_metrics, aggregate)
    write_dashboard(smart_summary, campaign_energy, shap_importance, shap_summary, aggregate)
    print("aggregate")
    print(aggregate.T)
    print("folds")
    print(folds[["campaign", "RMSE_W", "MAE_W", "RMSLE", "R2", "delta_R2"]])
    print("smartcity")
    print(smart_summary)
    print("shap top")
    print(shap_importance.head(10))
    print("c7 affine adaptation")
    print(c7_adaptation)
    print("c7 trajectory shift")
    print(c7_shift.head(12))


if __name__ == "__main__":
    main()
