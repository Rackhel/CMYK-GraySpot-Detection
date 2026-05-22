"""
scripts/generate_phase2_report.py

Phase 2 학습 결과를 시각화하는 HTML 리포트 생성 스크립트.
Generates an HTML report visualizing Phase 2 training results.

목적 / Purpose:
    run_phase2.py 완료 후 생성된 체크포인트와 히스토리 CSV를 읽어
    6개 탭으로 구성된 통합 HTML 리포트를 생성한다.
    Reads checkpoints and history CSVs after run_phase2.py completes,
    and generates an integrated HTML report with 6 tabs.

실행 방법 / How to run:
    python src/scripts/generate_phase2_report.py
    python src/scripts/generate_phase2_report.py --tag v1
    python src/scripts/generate_phase2_report.py --channels Y M --no-inference
    python src/scripts/generate_phase2_report.py --open-browser

전제 조건 / Prerequisites:
    - run_phase2.py 실행 완료
      -> outputs/checkpoints/phase2_summary_{tag}.json
      -> outputs/checkpoints/phase2_history_{ch}.csv
      -> outputs/checkpoints/phase2_{ch}_{backbone}_{tag}.pt  (또는 best_{ch}.pt)

Python 3.11.5 | macOS & Windows compatible
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.confusion import plot_confusion_matrix
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    NUM_LEVELS,
    TARGET_MAE,
    TARGET_OVERALL_ACC,
    TARGET_PER_CLASS_F1,
    TARGET_PER_COLOR_ACC,
    check_targets,
    compute_all_channels,
)
from src.utils import (
    build_model,
    create_directories,
    get_logger,
    get_nested,
    load_config,
    setup_logging,
    validate_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CKPT_DIR = ROOT / "outputs" / "checkpoints"
REPORT_DIR = ROOT / "outputs" / "reports"


# ---------------------------------------------------------------------------
# Helper: auto-detect backbone / hidden_dim from checkpoint weights
# ---------------------------------------------------------------------------


def _cfg_for_ckpt(cfg: dict, ckpt: Path) -> dict:
    """
    체크포인트 weight shape에서 backbone과 hidden_dim을 역산해 cfg 복사본을 패치한다.
    Auto-detects backbone and hidden_dim from checkpoint weight shapes.
    """
    import torch

    _FEATURE_TO_BACKBONE: dict = {
        2048: "resnet50",
        1280: "efficientnet_b0",
        1792: "efficientnet_b4",
        1536: "efficientnet_b3",
        1408: "efficientnet_b2",
        512: "resnet18",
        1024: "densenet121",
    }

    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    w0 = state.get("head.net.0.weight")
    if w0 is None:
        return cfg

    in_features = int(w0.shape[1])
    first_out = int(w0.shape[0])

    num_classes = cfg.get("data", {}).get("num_levels", 6)
    w4 = state.get("head.net.4.weight")
    if w4 is not None and int(w4.shape[0]) == num_classes:
        mid_dim = None
        hidden_dim = first_out
    elif w4 is not None:
        mid_dim = first_out
        hidden_dim = int(w4.shape[1])
    else:
        mid_dim = None
        hidden_dim = first_out

    patched = copy.deepcopy(cfg)

    detected_backbone = _FEATURE_TO_BACKBONE.get(in_features)
    if detected_backbone is not None:
        patched.setdefault("model", {})["backbone"] = detected_backbone
        backbone = detected_backbone
    else:
        backbone = patched.get("model", {}).get("backbone", "efficientnet_b0")

    heads = patched.setdefault("phase2", {}).setdefault("heads", {})
    if backbone not in heads:
        heads[backbone] = {"dropout": 0.3}
    heads[backbone]["mid_dim"] = mid_dim
    heads[backbone]["hidden_dim"] = hidden_dim
    return patched


# ---------------------------------------------------------------------------
# Summary loading
# ---------------------------------------------------------------------------


def find_latest_phase2_summary() -> Path | None:
    """outputs/checkpoints/ 에서 가장 최근 phase2_summary_*.json 파일을 찾는다."""
    candidates = sorted(CKPT_DIR.glob("phase2_summary_*.json"))
    return candidates[-1] if candidates else None


def load_phase2_summary(tag: str | None) -> tuple[dict, str]:
    """
    phase2_summary_{tag}.json 을 로드하고 (summary_dict, tag) 를 반환한다.
    """
    if tag:
        path = CKPT_DIR / f"phase2_summary_{tag}.json"
    else:
        path = find_latest_phase2_summary()
        if path is None:
            raise FileNotFoundError(
                f"No phase2_summary_*.json found in {CKPT_DIR}.\n"
                "Run run_phase2.py first."
            )

    if not path.exists():
        raise FileNotFoundError(
            f"phase2_summary not found: {path}\nRun run_phase2.py first."
        )

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    detected_tag = summary.get("cycle_tag") or path.stem.replace("phase2_summary_", "")
    return summary, detected_tag


def summary_to_cards(summary: dict, available: list) -> dict:
    """phase2_summary 채널별 결과를 성능 카드 dict 로 변환한다."""
    cards = {}
    valid = [
        r
        for r in summary["results"]
        if not r.get("skipped") and r["channel"] in available
    ]

    for r in valid:
        cards[r["channel"]] = {
            "test_acc": r["test_acc"],
            "mae": r["mae"],
            "best_val_acc": r["best_val_acc"],
            "n_train": r.get("n_train", 0),
            "n_val": r.get("n_val", 0),
            "n_test": r.get("n_test", 0),
            "pass_acc": r.get("pass_acc", r["test_acc"] >= TARGET_PER_COLOR_ACC),
            "pass_mae": r.get("pass_mae", r["mae"] <= TARGET_MAE),
        }

    if valid:
        cards["overall"] = {
            "test_acc": round(sum(r["test_acc"] for r in valid) / len(valid), 4),
            "mae": round(sum(r["mae"] for r in valid) / len(valid), 4),
        }

    return cards


# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------


def find_checkpoint(ch: str, backbone: str, tag: str) -> Path | None:
    """
    체크포인트 경로 탐색 순서:
    1. outputs/checkpoints/phase2_{ch}_{backbone}_{tag}.pt
    2. outputs/checkpoints/best_{ch}.pt
    3. None (추론 skip)
    """
    # Normalize backbone name: efficientnet_b0 -> effb0
    short = (
        backbone.replace("efficientnet_b", "effb")
        .replace("resnet", "res")
        .replace("_", "")
    )
    p1 = CKPT_DIR / f"phase2_{ch}_{short}_{tag}.pt"
    if p1.exists():
        return p1
    p2 = CKPT_DIR / f"best_{ch}.pt"
    if p2.exists():
        return p2
    return None


# ---------------------------------------------------------------------------
# History CSV loading
# ---------------------------------------------------------------------------


def load_history(ch: str) -> pd.DataFrame | None:
    """phase2_history_{ch}.csv 를 로드한다. 없으면 None."""
    path = CKPT_DIR / f"phase2_history_{ch}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------


def _build_training_curves(channels: list[str], max_epochs: int) -> dict[str, object]:
    """
    채널별 Training Curves (loss + accuracy).
    Each channel -> dict with 'loss_div' and 'acc_div' HTML strings.
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    figs = {}
    for ch in channels:
        df = load_history(ch)
        if df is None or df.empty:
            figs[ch] = None
            continue

        n_epochs = len(df)
        early_stop = n_epochs < max_epochs

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"[{ch}] Loss", f"[{ch}] Accuracy"],
            horizontal_spacing=0.12,
        )

        # Loss traces
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["train_loss"],
                name="train_loss",
                line=dict(color="#66d9ff", width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_loss"],
                name="val_loss",
                line=dict(color="#ff7aa2", width=2, dash="dash"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Accuracy traces
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["train_acc"],
                name="train_acc",
                line=dict(color="#50e3c2", width=2),
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_acc"],
                name="val_acc",
                line=dict(color="#c792ea", width=2, dash="dash"),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        # Early stopping vertical line
        if early_stop:
            for col in [1, 2]:
                fig.add_vline(
                    x=n_epochs,
                    line_dash="dot",
                    line_color="#f39c12",
                    line_width=1.5,
                    annotation_text=f"Early stop @ {n_epochs}",
                    annotation_font_color="#f39c12",
                    annotation_font_size=10,
                    row=1,
                    col=col,
                )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,18,32,0.6)",
            font=dict(family="Segoe UI", size=12, color="#e6eef8"),
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        figs[ch] = pio.to_html(
            fig, full_html=False, include_plotlyjs=False, div_id=f"fig-curve-{ch}"
        )
    return figs


def _build_per_class_bars(metrics: dict, available: list) -> dict[str, object]:
    """채널별 Per-Class F1/Precision/Recall 바 차트."""
    import plotly.graph_objects as go
    import plotly.io as pio

    figs = {}
    for ch in available:
        if ch not in metrics:
            figs[ch] = None
            continue
        pc = metrics[ch].get("per_class", [])
        if not pc:
            figs[ch] = None
            continue

        levels = [f"Level {p['level']}" for p in pc]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="F1", x=levels, y=[p["f1"] for p in pc], marker_color="#1976D2")
        )
        fig.add_trace(
            go.Bar(
                name="Precision",
                x=levels,
                y=[p["precision"] for p in pc],
                marker_color="#388E3C",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Recall",
                x=levels,
                y=[p["recall"] for p in pc],
                marker_color="#F57C00",
            )
        )
        fig.add_hline(
            y=TARGET_PER_CLASS_F1,
            line_dash="dash",
            line_color="#ff7aa2",
            annotation_text=f"F1 Target >= {TARGET_PER_CLASS_F1:.2f}",
        )

        fig.update_layout(
            title=f"[{ch}] Per-Class F1 / Precision / Recall",
            barmode="group",
            yaxis=dict(range=[0, 1.15], title="Score"),
            xaxis=dict(title="Level"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,18,32,0.6)",
            font=dict(family="Segoe UI", size=12, color="#e6eef8"),
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        figs[ch] = pio.to_html(
            fig, full_html=False, include_plotlyjs=False, div_id=f"fig-f1-{ch}"
        )
    return figs


def _build_mae_heatmap(all_results: dict, channels: list) -> object:
    """MAE Heatmap (Color x True Level)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    level_names = [f"Level {i}" for i in range(NUM_LEVELS)]
    mae_matrix = np.full((len(channels), NUM_LEVELS), np.nan)
    cnt_matrix = np.zeros((len(channels), NUM_LEVELS), dtype=int)

    for ci, ch in enumerate(channels):
        if ch not in all_results:
            continue
        yt = all_results[ch]["y_true"]
        yp = all_results[ch]["y_pred"]
        for lv in range(NUM_LEVELS):
            mask = yt == lv
            if mask.sum() > 0:
                mae_matrix[ci, lv] = float(
                    np.mean(np.abs(yt[mask].astype(float) - yp[mask].astype(float)))
                )
                cnt_matrix[ci, lv] = int(mask.sum())

    annot = [
        [
            (
                f"{mae_matrix[r, c]:.2f}<br>(n={cnt_matrix[r, c]})"
                if not np.isnan(mae_matrix[r, c])
                else "N/A"
            )
            for c in range(NUM_LEVELS)
        ]
        for r in range(len(channels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=mae_matrix,
            x=level_names,
            y=channels,
            text=annot,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=0,
            zmax=2.0,
            colorbar=dict(title="MAE", thickness=14),
        )
    )
    fig.update_layout(
        title=f"MAE per (Color x True Level) — Target <= {TARGET_MAE}",
        xaxis=dict(title="True Level"),
        yaxis=dict(title="Channel"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.6)",
        font=dict(family="Segoe UI", size=12, color="#e6eef8"),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False, div_id="fig-mae-heat"
    )


def _build_mae_dist(all_results: dict, channels: list) -> object:
    """예측 오차(pred - true) 분포 히스토그램."""
    import plotly.graph_objects as go
    import plotly.io as pio

    fig = go.Figure()
    colors = {"Y": "#f5e642", "M": "#e91e8c", "C": "#00b4d8", "K": "#cccccc"}
    for ch in channels:
        if ch not in all_results:
            continue
        yt = all_results[ch]["y_true"].astype(float)
        yp = all_results[ch]["y_pred"].astype(float)
        err = yp - yt
        fig.add_trace(
            go.Histogram(
                x=err,
                name=ch,
                marker_color=colors.get(ch, "#aaa"),
                opacity=0.7,
                xbins=dict(start=-5.5, end=5.5, size=1),
            )
        )
    fig.update_layout(
        title="Prediction Error Distribution (Pred - True)",
        barmode="overlay",
        xaxis=dict(title="Error (Pred - True)", dtick=1),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.6)",
        font=dict(family="Segoe UI", size=12, color="#e6eef8"),
        height=340,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False, div_id="fig-mae-dist"
    )


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#0b1220;--sf:#111927;--sf2:#172032;--bd:rgba(255,255,255,0.08);
--fg:#e6eef8;--fm:#9fb4c8;--c1:#66d9ff;--c2:#50e3c2;
--pass:#2ecc71;--fail:#e74c3c}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--fg);
line-height:1.7;padding:0;max-width:1400px;margin:0 auto}
.rh{background:linear-gradient(135deg,#0d1b2e,#0f2a40 60%,#0b1220);
border-bottom:2px solid rgba(102,217,255,.20);padding:2rem 2.5rem 1.6rem}
.rh h1{font-size:1.8rem;color:var(--c1);margin-bottom:.3rem}
.rh .meta{color:rgba(230,238,248,.5);font-size:.82rem;font-family:monospace;margin-top:.4rem}
.nav-tabs{display:flex;gap:0;background:var(--sf);border-bottom:1px solid var(--bd);
padding:0 2.5rem;position:sticky;top:0;z-index:100}
.nav-tab{padding:.75rem 1.2rem;cursor:pointer;color:rgba(230,238,248,.5);
font-size:.8rem;font-weight:600;letter-spacing:.04em;border-bottom:2px solid transparent;
transition:color .2s,border-color .2s;user-select:none;text-transform:uppercase}
.nav-tab:hover{color:var(--fg)}
.nav-tab.active{color:var(--c1);border-bottom-color:var(--c1)}
.section{display:none;padding:1.8rem 2.5rem}
.section.active{display:block}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:8px;
padding:1.4rem 1.6rem;margin-bottom:1.4rem}
.card h2{color:var(--c2);font-size:1rem;font-weight:700;margin-bottom:1rem;
padding-bottom:.5rem;border-bottom:1px solid var(--bd)}
.badge{display:inline-block;padding:.1rem .45rem;border-radius:4px;
font-size:.75rem;font-weight:700;margin-left:.35rem;vertical-align:middle}
.badge.pass{background:rgba(46,204,113,.18);color:var(--pass);border:1px solid var(--pass)}
.badge.fail{background:rgba(231,76,60,.18);color:var(--fail);border:1px solid var(--fail)}
.ch-badge{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.82rem;font-weight:700}
.ch-Y{background:rgba(245,230,66,.15);color:#f5e642;border:1px solid #f5e64280}
.ch-M{background:rgba(233,30,140,.15);color:#e91e8c;border:1px solid #e91e8c80}
.ch-C{background:rgba(0,180,216,.15);color:#00b4d8;border:1px solid #00b4d880}
.ch-K{background:rgba(180,180,180,.15);color:#ccc;border:1px solid #cccccc80}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
gap:1rem;margin-bottom:1.4rem}
.kpi-card{background:var(--sf);border:1px solid var(--bd);border-radius:8px;
padding:1rem 1.2rem;text-align:center;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.kpi-card.pass-card::before{background:var(--pass)}
.kpi-card.fail-card::before{background:var(--fail)}
.kpi-card .lbl{font-size:.7rem;color:var(--fm);letter-spacing:.06em;text-transform:uppercase;margin-bottom:.3rem}
.kpi-card .val{font-size:1.7rem;font-weight:700;font-family:monospace;line-height:1.1}
.kpi-card.pass-card .val{color:var(--pass)}.kpi-card.fail-card .val{color:var(--fail)}
.kpi-card .tgt{font-size:.68rem;color:var(--fm);margin-top:.3rem}
.kpi-card .delta{font-size:.78rem;margin-top:.2rem}
.delta-pos{color:var(--pass)}.delta-neg{color:var(--fail)}.delta-neu{color:var(--fm)}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{border:1px solid var(--bd);padding:.45rem .75rem;text-align:left}
th{background:var(--sf2);color:var(--fm);font-weight:600}
tr:hover{background:rgba(255,255,255,.02)}
.ch-tabs{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
.ch-tab{padding:.3rem .8rem;border-radius:4px;cursor:pointer;font-size:.82rem;
font-weight:600;border:1px solid var(--bd);color:var(--fm);background:var(--sf2)}
.ch-tab.active{color:var(--c1);border-color:var(--c1);background:rgba(102,217,255,.08)}
.ch-panel{display:none}.ch-panel.active{display:block}
.no-data{color:var(--fm);font-style:italic;padding:1rem;text-align:center;
background:var(--sf2);border-radius:6px}
.p3box{background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.4);
border-radius:6px;padding:1rem 1.2rem;font-size:.9rem;line-height:1.9}
.p3box strong{color:var(--pass);font-size:1rem}
.p3box-fail{background:rgba(231,76,60,.07);border:1px solid rgba(231,76,60,.3)}
.cmg{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:1rem}
.rf{text-align:center;color:var(--fm);font-size:.78rem;margin-top:2rem;
padding-top:1rem;border-top:1px solid var(--bd)}
"""

_TAB_JS = """
function switchTab(tabId) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    var tabEl = document.querySelector('.nav-tab[data-tab="' + tabId + '"]');
    var secEl = document.getElementById('sec-' + tabId);
    if (tabEl) tabEl.classList.add('active');
    if (secEl) secEl.classList.add('active');
}
function switchChTab(groupId, ch) {
    document.querySelectorAll('#' + groupId + ' .ch-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('#' + groupId + ' .ch-panel').forEach(p => p.classList.remove('active'));
    var tabEl = document.querySelector('#' + groupId + ' .ch-tab[data-ch="' + ch + '"]');
    var panEl = document.getElementById(groupId + '-panel-' + ch);
    if (tabEl) tabEl.classList.add('active');
    if (panEl) panEl.classList.add('active');
}
"""


def _badge(passed: bool) -> str:
    cls, txt = ("pass", "PASS") if passed else ("fail", "FAIL")
    return f'<span class="badge {cls}">{txt}</span>'


def _delta_html(d: float) -> str:
    if abs(d) < 0.0001:
        return f'<div class="delta delta-neu">vs baseline: 0.0000</div>'
    cls = "delta-pos" if d > 0 else "delta-neg"
    sign = "+" if d > 0 else ""
    return f'<div class="delta {cls}">vs baseline: {sign}{d:.4f}</div>'


def build_phase2_html(
    report_data: dict,
    output_path: Path,
) -> None:
    """6탭 통합 Phase 2 HTML 리포트를 생성한다."""
    meta = report_data["meta"]
    cards = report_data["cards"]
    metrics = report_data["metrics"]
    figs = report_data["figures"]
    phase3 = report_data["phase3_text"]
    available = report_data["available"]
    baseline_cards = report_data.get("baseline_cards", {})
    max_epochs = meta.get("epochs", 50)

    # ── Tab 1: Summary ──────────────────────────────────────────────────────
    color_rows = ""
    for ch in ["Y", "M", "C", "K"]:
        if ch not in cards:
            continue
        c = cards[ch]
        bc = baseline_cards.get(ch)
        delta_acc = f"<br>{_delta_html(c['test_acc'] - bc['test_acc'])}" if bc else ""
        delta_mae = f"<br>{_delta_html(-(c['mae'] - bc['mae']))}" if bc else ""
        color_rows += f"""
        <tr>
          <td><span class="ch-badge ch-{ch}">{ch}</span></td>
          <td>{c['test_acc']:.4f} {_badge(c['pass_acc'])}{delta_acc}</td>
          <td>{c['mae']:.4f} {_badge(c['pass_mae'])}{delta_mae}</td>
          <td>{c['best_val_acc']:.4f}</td>
          <td>{c['n_train']}</td><td>{c['n_val']}</td><td>{c['n_test']}</td>
        </tr>"""

    overall = cards.get("overall", {})
    avg_acc = overall.get("test_acc", 0)
    avg_mae = overall.get("mae", 0)
    n_total = sum(cards[c]["n_test"] for c in ["Y", "M", "C", "K"] if c in cards)
    acc_cls = "pass-card" if avg_acc >= TARGET_OVERALL_ACC else "fail-card"
    mae_cls = "pass-card" if avg_mae <= TARGET_MAE else "fail-card"

    sec_summary = f"""
    <div class="kpi-grid">
      <div class="kpi-card {acc_cls}">
        <div class="lbl">Avg. Test Accuracy</div>
        <div class="val">{avg_acc:.1%}</div>
        <div class="tgt">Target >= {TARGET_OVERALL_ACC:.0%}</div>
      </div>
      <div class="kpi-card {mae_cls}">
        <div class="lbl">Avg. MAE</div>
        <div class="val">{avg_mae:.4f}</div>
        <div class="tgt">Target <= {TARGET_MAE:.2f}</div>
      </div>
      <div class="kpi-card pass-card">
        <div class="lbl">Test Samples</div>
        <div class="val">{n_total}</div>
        <div class="tgt">Total</div>
      </div>
    </div>
    <div class="card">
      <h2>Channel Performance (Test-set Official — phase2_summary)</h2>
      <table>
        <thead><tr>
          <th>Channel</th>
          <th>Test Acc (target>={TARGET_PER_COLOR_ACC:.0%})</th>
          <th>MAE (target<={TARGET_MAE:.2f})</th>
          <th>Best Val Acc</th>
          <th>N Train</th><th>N Val</th><th>N Test</th>
        </tr></thead>
        <tbody>{color_rows}</tbody>
      </table>
    </div>"""

    # ── Tab 2: Training Curves ──────────────────────────────────────────────
    curve_figs = figs.get("training_curves", {})
    # Build channel-tab UI
    first_ch = available[0] if available else "Y"
    ch_tabs_html = "".join(
        f'<div class="ch-tab{"active" if ch == first_ch else ""}" '
        f"data-ch=\"{ch}\" onclick=\"switchChTab('curves','{ch}')\">{ch}</div>"
        for ch in available
    )
    ch_panels_html = ""
    for ch in available:
        active_cls = "active" if ch == first_ch else ""
        curve_div = curve_figs.get(ch)
        if curve_div:
            content = curve_div
        else:
            content = (
                f'<div class="no-data">No history CSV data for channel {ch}.</div>'
            )
        ch_panels_html += (
            f'<div class="ch-panel {active_cls}" id="curves-panel-{ch}">{content}</div>'
        )

    sec_curves = f"""
    <div class="card">
      <h2>Training Curves — Loss &amp; Accuracy by Epoch</h2>
      <div class="ch-tabs" id="curves">{ch_tabs_html}</div>
      {ch_panels_html}
    </div>"""

    # ── Tab 3: Confusion Matrix ─────────────────────────────────────────────
    cm_figs = figs.get("confusion", {})
    cm_divs = ""
    for ch in available:
        key = f"cm_{ch}"
        if key in cm_figs:
            cm_divs += cm_figs[key]
    overall_cm = cm_figs.get("cm_overall", "")

    sec_cm = f"""
    <div class="card">
      <h2>Confusion Matrix (Row-Normalized) — Per Channel</h2>
      <div class="cmg">{cm_divs}</div>
    </div>
    <div class="card">
      <h2>Confusion Matrix — Overall</h2>
      {overall_cm if overall_cm else '<div class="no-data">Overall CM not available (no inference).</div>'}
    </div>"""

    # ── Tab 4: Per-Class F1 ─────────────────────────────────────────────────
    f1_figs = figs.get("per_class", {})
    first_ch_f1 = available[0] if available else "Y"
    ch_tabs_f1 = "".join(
        f'<div class="ch-tab{"active" if ch == first_ch_f1 else ""}" '
        f"data-ch=\"{ch}\" onclick=\"switchChTab('f1tabs','{ch}')\">{ch}</div>"
        for ch in available
    )
    ch_panels_f1 = ""
    for ch in available:
        active_cls = "active" if ch == first_ch_f1 else ""
        fig_div = f1_figs.get(ch)
        if fig_div:
            content = fig_div
        else:
            content = f'<div class="no-data">No per-class data for channel {ch}.</div>'
        ch_panels_f1 += (
            f'<div class="ch-panel {active_cls}" id="f1tabs-panel-{ch}">{content}</div>'
        )

    # Per-class table (overall metrics if available)
    pc_rows = ""
    if "overall" in metrics:
        for pc in metrics["overall"].get("per_class", []):
            pc_pass = pc["f1"] >= TARGET_PER_CLASS_F1
            pc_rows += f"""
            <tr>
              <td>Level {pc['level']}</td>
              <td>{pc['precision']:.4f}</td>
              <td>{pc['recall']:.4f}</td>
              <td>{pc['f1']:.4f} {_badge(pc_pass)}</td>
            </tr>"""

    sec_f1 = f"""
    <div class="card">
      <h2>Per-Class F1 / Precision / Recall — by Channel</h2>
      <div class="ch-tabs" id="f1tabs">{ch_tabs_f1}</div>
      {ch_panels_f1}
    </div>"""
    if pc_rows:
        sec_f1 += f"""
    <div class="card">
      <h2>Overall Per-Level Summary</h2>
      <table>
        <thead><tr><th>Level</th><th>Precision</th><th>Recall</th>
        <th>F1 (target>={TARGET_PER_CLASS_F1:.2f})</th></tr></thead>
        <tbody>{pc_rows}</tbody>
      </table>
    </div>"""

    # ── Tab 5: MAE ──────────────────────────────────────────────────────────
    mae_heat = figs.get("mae_heatmap", "")
    mae_dist = figs.get("mae_dist", "")

    sec_mae = f"""
    <div class="card">
      <h2>MAE Heatmap (Channel x True Level)</h2>
      {mae_heat if mae_heat else '<div class="no-data">MAE heatmap not available (no inference).</div>'}
    </div>
    <div class="card">
      <h2>Prediction Error Distribution</h2>
      {mae_dist if mae_dist else '<div class="no-data">Error distribution not available (no inference).</div>'}
    </div>"""

    # ── Tab 6: Phase 3 Feedback ─────────────────────────────────────────────
    all_pass = all(cards[c]["pass_acc"] and cards[c]["pass_mae"] for c in available)
    box_cls = "p3box" if all_pass else "p3box p3box-fail"
    phase3_html = (
        phase3.replace("\n", "<br>")
        .replace("TERMINATE", "<strong>TERMINATE</strong>")
        .replace("PASS", "<strong>PASS</strong>")
    )
    sec_p3 = f"""
    <div class="card">
      <h2>Phase 3 Feedback Decision (PRD 3.3.2)</h2>
      <div class="{box_cls}">{phase3_html}</div>
    </div>"""

    # ── Tabs list ───────────────────────────────────────────────────────────
    tabs = [
        ("summary", "1. Summary"),
        ("curves", "2. Training Curves"),
        ("cm", "3. Confusion Matrix"),
        ("f1", "4. Per-Class F1"),
        ("mae", "5. MAE"),
        ("phase3", "6. Phase 3 Feedback"),
    ]
    tabs_html = "".join(
        f'<div class="nav-tab{"active" if i == 0 else ""}" '
        f'data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</div>'
        for i, (tid, label) in enumerate(tabs)
    )

    sections_map = {
        "summary": sec_summary,
        "curves": sec_curves,
        "cm": sec_cm,
        "f1": sec_f1,
        "mae": sec_mae,
        "phase3": sec_p3,
    }
    sections_html = "".join(
        f'<section id="sec-{tid}" class="section{"active" if i == 0 else ""}">'
        f"{sections_map[tid]}</section>"
        for i, (tid, _) in enumerate(tabs)
    )

    generated_at = meta["generated_at"]
    tag = meta["tag"]
    backbone = meta["backbone"]
    epochs = meta["epochs"]
    image_size = meta["image_size"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Phase 2 Report — {tag}</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body>

<div class="rh">
  <h1>Grayspot Phase 2 Evaluation Report</h1>
  <div style="margin-top:.5rem">
    {'<span style="color:#2ecc71">&#10003; All targets met</span>' if all_pass else '<span style="color:#e74c3c">&#10007; Below target</span>'}
  </div>
  <div class="meta">
    Tag: {tag} &nbsp;|&nbsp; Generated: {generated_at} &nbsp;|&nbsp;
    Backbone: {backbone} &nbsp;|&nbsp; Epochs: {epochs} &nbsp;|&nbsp;
    Image: {image_size}x{image_size} &nbsp;|&nbsp;
    Channels: {', '.join(available)}
  </div>
</div>

<nav class="nav-tabs">{tabs_html}</nav>
<main>{sections_html}</main>

<div class="rf">
  Grayspot Detection Pipeline &middot; Phase 2 Report &middot; {generated_at}
</div>

<script>{_TAB_JS}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 evaluation HTML report"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Cycle tag (e.g. v1). Defaults to latest phase2_summary file.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to include (default: all from summary)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip checkpoint loading; use history/summary only",
    )
    args = parser.parse_args()

    cfg = load_config()
    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        sys.exit(1)

    create_directories(cfg)
    setup_logging(
        log_dir=Path(cfg["storage"]["logs_dir"]),
        level=get_nested(cfg, "logging.level") or "INFO",
        format_style=get_nested(cfg, "logging.format") or "detailed",
        console=get_nested(cfg, "logging.console_output"),
        file=get_nested(cfg, "logging.file_output"),
    )
    logger = get_logger(__name__)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 2 Report Generation")
    logger.info(f"  Tag          : {args.tag or 'auto-detect'}")
    logger.info(f"  No-inference : {args.no_inference}")
    logger.info("=" * 60)

    # ── 1. Load phase2_summary ────────────────────────────────────────────
    summary, tag = load_phase2_summary(args.tag)
    logger.info(f"Loaded phase2_summary (tag={tag})")

    backbone = summary.get("backbone", cfg["model"]["backbone"])
    epochs = summary.get("epochs", cfg["phase2"]["epochs"])

    all_results_in_summary = [
        r["channel"] for r in summary["results"] if not r.get("skipped")
    ]
    if args.channels:
        available = [c for c in args.channels if c in all_results_in_summary]
    else:
        available = all_results_in_summary

    if not available:
        logger.error("No valid channels found in phase2_summary.")
        sys.exit(1)

    cards = summary_to_cards(summary, available)
    logger.info(f"Channels: {available}")
    for ch in available:
        c = cards[ch]
        logger.info(
            f'  [{ch}] Test Acc: {c["test_acc"]:.4f} '
            f'{"[PASS]" if c["pass_acc"] else "[FAIL]"} | '
            f'MAE: {c["mae"]:.4f} '
            f'{"[PASS]" if c["pass_mae"] else "[FAIL]"}'
        )

    # ── 2. Baseline comparison (optional) ────────────────────────────────
    baseline_cards: dict = {}
    baseline_summary_path = (
        Path(cfg["storage"]["data_root"]) / "baseline" / "baseline_summary.json"
    )
    if baseline_summary_path.exists():
        try:
            with open(baseline_summary_path, encoding="utf-8") as f:
                bl = json.load(f)
            for r in bl.get("results", []):
                if not r.get("skipped") and r["channel"] in available:
                    baseline_cards[r["channel"]] = {
                        "test_acc": r["test_acc"],
                        "mae": r["mae"],
                    }
            logger.info(f"Loaded baseline comparison for: {list(baseline_cards)}")
        except Exception as e:
            logger.warning(f"Could not load baseline_summary: {e}")

    # ── 3. Inference (unless --no-inference) ─────────────────────────────
    all_results: dict = {}
    import torch

    device = torch.device(cfg["system"]["device"])

    if not args.no_inference:
        labeled_dir = Path(cfg["storage"]["labeled_dir"])
        labels_csv = Path(cfg["storage"]["data_root"]) / "labels_master.csv"
        tmp_dir = REPORT_DIR / "tmp"

        for ch in available:
            ckpt = find_checkpoint(ch, backbone, tag)
            if ckpt is None:
                logger.warning(f"[{ch}] No checkpoint found — skipping inference")
                continue
            logger.info(f"[{ch}] Inference from: {ckpt}")
            try:
                model = build_model(_cfg_for_ckpt(cfg, ckpt), ckpt, device)
                ev = Evaluator(
                    model=model,
                    labeled_dir=labeled_dir,
                    labels_csv=labels_csv,
                    output_dir=tmp_dir,
                    device=device,
                    image_size=cfg["data"]["image_size"],
                    batch_size=cfg["phase2"]["batch_size"],
                    num_levels=cfg["data"]["num_levels"],
                    cfg=cfg,
                )
                all_results.update(ev.run(channels=[ch]))
                logger.info(f"[{ch}] Inference done")
            except Exception as e:
                logger.warning(f"[{ch}] Inference failed: {e}")
    else:
        logger.info("--no-inference: skipping checkpoint loading")

    # ── 4. Compute metrics (from inference results or empty) ──────────────
    metrics: dict = {}
    if all_results:
        metrics = compute_all_channels(
            all_results,
            list(all_results.keys()),
            num_classes=cfg["data"]["num_levels"],
        )

    # ── 5. Build figures ──────────────────────────────────────────────────
    logger.info("Building figures...")
    import plotly.io as pio

    figs: dict = {}

    # Training curves
    figs["training_curves"] = _build_training_curves(available, max_epochs=epochs)

    # Per-class F1 bars (from metrics if available, else summary-derived stub)
    if metrics:
        figs["per_class"] = _build_per_class_bars(metrics, available)
    else:
        # Build stub per-class data from summary (no precision/recall available)
        figs["per_class"] = {ch: None for ch in available}

    # Confusion matrices
    cm_figs: dict = {}
    if all_results:
        for ch in all_results:
            yt, yp = all_results[ch]["y_true"], all_results[ch]["y_pred"]
            ch_acc = metrics[ch]["accuracy"] if ch in metrics else 0.0
            fig = plot_confusion_matrix(
                yt,
                yp,
                title=f"[{ch}] Confusion Matrix  Acc={ch_acc:.4f}",
                normalize=True,
            )
            cm_figs[f"cm_{ch}"] = pio.to_html(
                fig, full_html=False, include_plotlyjs=False, div_id=f"fig-cm-{ch}"
            )
        all_true = np.concatenate([all_results[c]["y_true"] for c in all_results])
        all_pred = np.concatenate([all_results[c]["y_pred"] for c in all_results])
        ov_acc = metrics.get("overall", {}).get("accuracy", 0.0)
        fig_ov = plot_confusion_matrix(
            all_true,
            all_pred,
            title=f"[Overall] Confusion Matrix  Acc={ov_acc:.4f}",
            normalize=True,
        )
        cm_figs["cm_overall"] = pio.to_html(
            fig_ov, full_html=False, include_plotlyjs=False, div_id="fig-cm-overall"
        )
    figs["confusion"] = cm_figs

    # MAE heatmap / distribution
    if all_results:
        figs["mae_heatmap"] = _build_mae_heatmap(all_results, list(all_results.keys()))
        figs["mae_dist"] = _build_mae_dist(all_results, list(all_results.keys()))
    else:
        figs["mae_heatmap"] = ""
        figs["mae_dist"] = ""

    # ── 6. Phase 3 feedback text ──────────────────────────────────────────
    lines = ["=== Phase 3 Feedback Decision (PRD 3.3.2) ==="]
    all_pass_flag = all(
        cards[c]["pass_acc"] and cards[c]["pass_mae"] for c in available
    )

    if all_pass_flag:
        lines.append("All targets met -- TERMINATE Swing")
    else:
        lines.append("Action required:")
        for ch in available:
            c = cards[ch]
            if not c["pass_acc"]:
                lines.append(
                    f'  [{ch}] Test Acc {c["test_acc"]:.3f} < {TARGET_PER_COLOR_ACC}'
                    " -> retry Phase 2 or Phase 0 (representation)"
                )
            if not c["pass_mae"]:
                lines.append(
                    f'  [{ch}] MAE {c["mae"]:.3f} > {TARGET_MAE}'
                    " -> Phase 0 (representation learning retry)"
                )

    lines += [
        "",
        f'  Avg. Test Accuracy : {cards["overall"]["test_acc"]:.4f}'
        f"  (target >= {TARGET_PER_COLOR_ACC})",
        f'  Avg. MAE           : {cards["overall"]["mae"]:.4f}'
        f"  (target <= {TARGET_MAE})",
        f"  Source             : phase2_summary_{tag}.json (Test-set 15%)",
    ]
    phase3_text = "\n".join(lines)

    # ── 7. Build HTML ─────────────────────────────────────────────────────
    meta = {
        "tag": tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": backbone,
        "epochs": epochs,
        "image_size": cfg["data"]["image_size"],
        "channels": available,
    }

    output_path = REPORT_DIR / f"phase2_{tag}.html"
    build_phase2_html(
        report_data={
            "meta": meta,
            "cards": cards,
            "metrics": metrics,
            "figures": figs,
            "phase3_text": phase3_text,
            "available": available,
            "baseline_cards": baseline_cards,
        },
        output_path=output_path,
    )

    logger.info("=" * 60)
    logger.info(f"Report complete -> {output_path}")
    logger.info("=" * 60)

    print("\n" + phase3_text)
    print(f"\nReport saved: {output_path}")

    if args.open_browser:
        import webbrowser

        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
