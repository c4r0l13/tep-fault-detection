"""End-to-end experiment: detection (PCA vs DPCA), root-cause hints, diagnosis.

Run with ``python -m tep_fdd.pipeline --data data/raw --out reports``.
Writes ``reports/metrics.json``, ``reports/results.md`` and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from . import style
from .data import (
    FAULT_DESCRIPTIONS,
    FAULT_ONSET_TEST,
    N_FAULTS,
    SAMPLE_MINUTES,
    VARIABLE_CODES,
    VARIABLE_NAMES,
    load_run,
    load_tep,
)
from .diagnosis import make_models, window_features
from .metrics import detection_delay, false_alarm_rate, fault_detection_rate
from .monitoring import PCAMonitor

SEED = 42


def run_detection(data_dir: Path) -> tuple[pd.DataFrame, dict[str, PCAMonitor]]:
    normal_train = load_run(data_dir, 0, "train")
    # Independent normal run used ONLY to calibrate control limits. Evaluation of
    # false alarms uses the pre-onset part of the 21 fault runs (other simulations).
    normal_cal = load_run(data_dir, 0, "test")
    q = 0.995  # per statistic -> ~1% combined false-alarm target
    monitors = {
        "PCA, theoretical limits": PCAMonitor(0.90, 0.01, lags=0).fit(normal_train),
        "PCA": PCAMonitor(0.90, 0.01, lags=0).fit(normal_train).calibrate(normal_cal, q),
        "DPCA (2 lags)": PCAMonitor(0.90, 0.01, lags=2).fit(normal_train).calibrate(normal_cal, q),
    }
    rows = []
    for name, mon in monitors.items():
        for f in range(1, N_FAULTS + 1):
            X = load_run(data_dir, f, "test")
            alarms = mon.predict(X)
            delay = detection_delay(alarms, FAULT_ONSET_TEST, consecutive=6)
            rows.append(
                {
                    "model": name,
                    "fault": f,
                    "description": FAULT_DESCRIPTIONS[f],
                    "fdr": fault_detection_rate(alarms, FAULT_ONSET_TEST),
                    "far_pre_onset": false_alarm_rate(alarms, FAULT_ONSET_TEST),
                    "delay_min": None if delay is None else delay * SAMPLE_MINUTES,
                    "n_components": mon.n_components_,
                }
            )
    return pd.DataFrame(rows), monitors


def run_diagnosis(data_dir: Path) -> dict:
    data = load_tep(data_dir)
    run_len_train = np.bincount(data.y_train)  # one run per class in train
    g_train = np.repeat(np.arange(N_FAULTS + 1), run_len_train)
    g_test = np.repeat(np.arange(N_FAULTS + 1), 960)

    results: dict = {}
    for name, model in make_models(SEED).items():
        if "window" in name:
            Xtr = window_features(data.X_train, g_train)
            Xte = window_features(data.X_test, g_test)
        else:
            Xtr, Xte = data.X_train, data.X_test
        t0 = time.perf_counter()
        model.fit(Xtr, data.y_train)
        pred = model.predict(Xte)
        results[name] = {
            "accuracy": float(accuracy_score(data.y_test, pred)),
            "macro_f1": float(f1_score(data.y_test, pred, average="macro")),
            "f1_per_class": f1_score(data.y_test, pred, average=None).tolist(),
            "recall_per_class": recall_score(data.y_test, pred, average=None).tolist(),
            "macro_f1_excl_3_9_15": float(
                f1_score(
                    data.y_test,
                    pred,
                    average="macro",
                    labels=[c for c in range(N_FAULTS + 1) if c not in (3, 9, 15)],
                )
            ),
            "fit_seconds": round(time.perf_counter() - t0, 1),
            "_pred": pred,
        }
    results["_y_true"] = data.y_test
    return results


# --------------------------------------------------------------------------- figures


def fig_detection(det: pd.DataFrame, out: Path) -> None:
    piv = det.pivot(index="fault", columns="model", values="fdr")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(piv))
    w = 0.38
    for i, col in enumerate(["PCA", "DPCA (2 lags)"]):
        ax.bar(x + (i - 0.5) * w, piv[col] * 100, width=w - 0.04, label=col, color=style.SERIES[i])
    ax.set_xticks(x, [f"IDV{f}" for f in piv.index], rotation=45, ha="right")
    ax.set_ylabel("Fault detection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Detection rate per fault: most faults are caught, IDV3, 9 and 15 are not")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "detection_rate_per_fault.png")
    plt.close(fig)


def fig_control_chart(mon: PCAMonitor, data_dir: Path, fault: int, out: Path) -> None:
    X = load_run(data_dir, fault, "test")
    t2, spe = mon.statistics(X)
    hours = np.arange(len(X)) * SAMPLE_MINUTES / 60
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, stat, lim, label in (
        (axes[0], t2, mon.t2_limit_, "Hotelling T²"),
        (axes[1], spe, mon.spe_limit_, "SPE (Q)"),
    ):
        ax.plot(hours, stat, color=style.SERIES[0], lw=1.1)
        ax.axhline(lim, color=style.INK_2, ls="--", lw=1)
        ax.text(hours[-1], lim, " 99% limit", va="bottom", ha="right", color=style.INK_2, fontsize=8)
        ax.axvline(FAULT_ONSET_TEST * SAMPLE_MINUTES / 60, color=style.CRITICAL, lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(label)
    axes[0].text(
        FAULT_ONSET_TEST * SAMPLE_MINUTES / 60,
        axes[0].get_ylim()[1],
        " fault introduced",
        color=style.CRITICAL,
        fontsize=8,
        va="top",
    )
    axes[0].set_title(f"Control charts, IDV{fault}: {FAULT_DESCRIPTIONS[fault]}")
    axes[1].set_xlabel("Time (hours)")
    fig.savefig(out / f"control_chart_idv{fault}.png")
    plt.close(fig)


def fig_contributions(mon: PCAMonitor, data_dir: Path, fault: int, out: Path, top: int = 10) -> list:
    X = load_run(data_dir, fault, "test")
    c = mon.spe_contributions(X)[FAULT_ONSET_TEST:].mean(axis=0)
    c = c / c.sum() * 100
    order = np.argsort(c)[::-1][:top][::-1]
    labels = [f"{VARIABLE_CODES[j]}  {VARIABLE_NAMES[j]}" for j in order]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [style.SERIES[0] if k == len(order) - 1 else "#86b6ef" for k in range(len(order))]
    ax.barh(labels, c[order], color=colors, height=0.7)
    for y, v in enumerate(c[order]):
        ax.text(v, y, f" {v:.0f}%", va="center", fontsize=8, color=style.INK_2)
    ax.set_xlabel("Share of SPE after fault onset (%)")
    ax.set_title(f"Root-cause hint for IDV{fault}: {FAULT_DESCRIPTIONS[fault]}")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / f"contributions_idv{fault}.png")
    plt.close(fig)
    return [(VARIABLE_CODES[j], VARIABLE_NAMES[j], round(float(c[j]), 1)) for j in order[::-1]]


def fig_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(N_FAULTS + 1), normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm * 100, cmap=style.BLUES, vmin=0, vmax=100)
    ticks = [str(i) for i in range(N_FAULTS + 1)]
    ax.set_xticks(range(N_FAULTS + 1), ticks)
    ax.set_yticks(range(N_FAULTS + 1), ticks)
    ax.set_xlabel("Predicted fault (0 = normal)")
    ax.set_ylabel("True fault (0 = normal)")
    ax.grid(False)
    for i in range(N_FAULTS + 1):
        v = cm[i, i] * 100
        ax.text(
            i, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else style.INK
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% of true class", color=style.INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Diagnosis confusion matrix (LightGBM, test runs)")
    fig.savefig(out / "confusion_matrix.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    style.apply_style()

    print("1/3 detection ...")
    det, monitors = run_detection(data_dir)
    fig_detection(det, figs)
    fig_control_chart(monitors["PCA"], data_dir, 1, figs)
    contrib = {f: fig_contributions(monitors["PCA"], data_dir, f, figs) for f in (4, 6)}

    print("2/3 diagnosis ...")
    diag = run_diagnosis(data_dir)
    y_true = diag.pop("_y_true")
    best = "LightGBM (window features)"
    fig_confusion(y_true, diag[best]["_pred"], figs)
    for v in diag.values():
        v.pop("_pred")

    print("3/3 writing reports ...")
    summary = (
        det.groupby("model")
        .agg(
            mean_fdr=("fdr", "mean"),
            false_alarm_rate=("far_pre_onset", "mean"),
            n_components=("n_components", "first"),
            faults_fdr_over_90=("fdr", lambda s: int((s > 0.9).sum())),
            median_delay_min=("delay_min", "median"),
        )
        .reset_index()
    )
    metrics = {
        "detection_summary": summary.to_dict(orient="records"),
        "detection_per_fault": det.to_dict(orient="records"),
        "contributions": {str(k): v for k, v in contrib.items()},
        "diagnosis": diag,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    det.to_csv(out / "detection_per_fault.csv", index=False)

    piv = det.pivot(index="fault", columns="model", values="fdr")[["PCA", "DPCA (2 lags)"]]
    piv = piv.mul(100).round(1)
    piv.insert(0, "Fault", [f"IDV{f}: {FAULT_DESCRIPTIONS[f]}" for f in piv.index])
    lines = ["# Results", "", "## Detection summary", "", summary.round(3).to_markdown(index=False), ""]
    lines += ["## Fault detection rate (%) per fault", "", piv.to_markdown(index=False), ""]
    lines += ["## Diagnosis", ""]
    for k, v in diag.items():
        lines.append(
            f"- **{k}**: accuracy {v['accuracy']:.3f}, macro-F1 {v['macro_f1']:.3f}, "
            f"macro-F1 excl. IDV3/9/15 {v['macro_f1_excl_3_9_15']:.3f}"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(summary.round(3).to_string(index=False))
    for k, v in diag.items():
        print(
            f"{k}: acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} excl={v['macro_f1_excl_3_9_15']:.3f}"
        )


if __name__ == "__main__":
    main()
