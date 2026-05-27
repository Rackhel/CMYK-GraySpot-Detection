"""
scripts/generate_optuna_report.py

Optuna 하이퍼파라미터 튜닝 결과를 시각화하는 통합 HTML 리포트 생성 스크립트.
Generates a unified HTML report visualizing Optuna hyperparameter tuning results.

탭 구성 / Tab layout:
    ① Summary        — KPI 카드: Accuracy / MAE / Macro-F1
    ② Optuna Params  — 채널별 best params 테이블 + 기본값 대비 변화량
    ③ Trial History  — 채널별 최적화 곡선 (trial number vs. objective value)
    ④ Confusion Matrix — 채널별 + Overall 혼동 행렬
    ⑤ Per-Class F1/MAE — 클래스별 F1·Precision·Recall 바 차트 + MAE 히트맵
    ⑥ Confidence     — 예측 신뢰도 분포

전제 조건 / Prerequisites:
    - run_optuna.py 실행 완료
      -> outputs/optuna/best_params_phase2_{ch}.json
      -> outputs/optuna/trials_summary_phase2_{ch}.json
      -> data_set/models/best_{ch}.pt

실행 방법 / How to run:
    python -m src.scripts.generate_optuna_report
    python -m src.scripts.generate_optuna_report --channels Y M
    python -m src.scripts.generate_optuna_report --open-browser

Python 3.11.5 | macOS & Windows compatible
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
    compute_all_channels,
)
from src.utils import build_model, load_config

OPTUNA_DIR = ROOT / "outputs" / "optuna"
MODELS_DIR = ROOT / "data_set" / "models"
REPORT_DIR = ROOT / "outputs" / "reports"

_ALL_CHANNELS = ["Y", "M", "C", "K"]

# ── CSS / JS (generate_phase2_report.py 와 동일한 디자인 시스템) ──────────────

_CSS = """
:root{--bg:#0b1220;--sf:#111927;--sf2:#172032;--bd:rgba(255,255,255,0.08);
--fg:#e6eef8;--fm:#9fb4c8;--c1:#66d9ff;--c2:#50e3c2;
--pass:#2ecc71;--fail:#e74c3c;--warn:#f39c12}
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
.badge.warn{background:rgba(243,156,18,.18);color:var(--warn);border:1px solid var(--warn)}
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
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{border:1px solid var(--bd);padding:.45rem .75rem;text-align:left}
th{background:var(--sf2);color:var(--fm);font-weight:600}
tr:hover{background:rgba(255,255,255,.02)}
td.changed{color:var(--warn);font-weight:600}
td.same{color:var(--fm)}
.ch-tabs{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
.ch-tab{padding:.3rem .8rem;border-radius:4px;cursor:pointer;font-size:.82rem;
font-weight:600;border:1px solid var(--bd);color:var(--fm);background:var(--sf2)}
.ch-tab.active{color:var(--c1);border-color:var(--c1);background:rgba(102,217,255,.08)}
.ch-panel{display:none}.ch-panel.active{display:block}
.no-data{color:var(--fm);font-style:italic;padding:1rem;text-align:center;
background:var(--sf2);border-radius:6px}
.cmg{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:1rem}
.rf{text-align:center;color:var(--fm);font-size:.78rem;margin-top:2rem;
padding-top:1rem;border-top:1px solid var(--bd)}
.param-highlight{background:rgba(243,156,18,.08);border-left:3px solid var(--warn)}
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


# ── 데이터 로더 / Data loaders ────────────────────────────────────────────────


def load_best_params(channel: str) -> dict | None:
    """outputs/optuna/best_params_phase2_{ch}.json 로드."""
    path = OPTUNA_DIR / f"best_params_phase2_{channel.lower()}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trials_summary(channel: str) -> list | None:
    """outputs/optuna/trials_summary_phase2_{ch}.json 로드."""
    path = OPTUNA_DIR / f"trials_summary_phase2_{channel.lower()}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_best_checkpoint(channel: str) -> Path | None:
    """data_set/models/best_{ch}.pt 탐색."""
    p = MODELS_DIR / f"best_{channel}.pt"
    return p if p.exists() else None


# ── Plotly figure builders ────────────────────────────────────────────────────


def _build_trial_history_figs(trials_by_channel: dict, available: list) -> dict:
    """
    채널별 Optuna 최적화 곡선.
    각 trial의 objective value와 누적 best value를 함께 표시한다.
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    figs = {}
    for ch in available:
        trials = trials_by_channel.get(ch)
        if not trials:
            figs[ch] = None
            continue

        completed = [t for t in trials if t.get("state") == "TrialState.COMPLETE"]
        if not completed:
            figs[ch] = None
            continue

        nums = [t["number"] for t in completed]
        vals = [t["value"] for t in completed]

        # 누적 best (maximize → 점진 최대, minimize → 점진 최소)
        direction = "maximize"  # Phase 2 기본값
        best_so_far = []
        cur = None
        for v in vals:
            if v is None:
                best_so_far.append(cur)
                continue
            if cur is None:
                cur = v
            else:
                cur = max(cur, v) if direction == "maximize" else min(cur, v)
            best_so_far.append(cur)

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(
            go.Scatter(
                x=nums,
                y=vals,
                mode="markers",
                name="Trial Value",
                marker=dict(color="#66d9ff", size=6, opacity=0.6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nums,
                y=best_so_far,
                mode="lines",
                name="Best So Far",
                line=dict(color="#50e3c2", width=2),
            )
        )
        fig.update_layout(
            title=f"[{ch}] Optuna Trial History (Phase 2 val_acc)",
            xaxis=dict(title="Trial #"),
            yaxis=dict(title="Val Accuracy"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,18,32,0.6)",
            font=dict(family="Segoe UI", size=12, color="#e6eef8"),
            height=360,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        figs[ch] = pio.to_html(
            fig, full_html=False, include_plotlyjs=False, div_id=f"fig-trial-{ch}"
        )
    return figs


def _build_per_class_bars(metrics: dict, available: list) -> dict:
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


def _build_mae_heatmap(all_results: dict, channels: list) -> str:
    """MAE Heatmap (Channel x True Level)."""
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
        title=f"MAE per (Channel x True Level) — Target <= {TARGET_MAE}",
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


def _build_conf_dist(all_results: dict, channels: list) -> str:
    """예측 신뢰도 분포 히스토그램."""
    import plotly.graph_objects as go
    import plotly.io as pio

    colors = {"Y": "#f5e642", "M": "#e91e8c", "C": "#00b4d8", "K": "#cccccc"}
    fig = go.Figure()
    for ch in channels:
        if ch not in all_results:
            continue
        confs = all_results[ch].get("confidences")
        if confs is None:
            continue
        fig.add_trace(
            go.Histogram(
                x=confs,
                name=ch,
                marker_color=colors.get(ch, "#aaa"),
                opacity=0.7,
                xbins=dict(start=0.0, end=1.0, size=0.05),
            )
        )
    fig.update_layout(
        title="Prediction Confidence Distribution",
        barmode="overlay",
        xaxis=dict(title="Confidence", range=[0, 1]),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.6)",
        font=dict(family="Segoe UI", size=12, color="#e6eef8"),
        height=340,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id="fig-conf")


# ── HTML section builders ─────────────────────────────────────────────────────


def _badge(passed: bool) -> str:
    cls, txt = ("pass", "PASS") if passed else ("fail", "FAIL")
    return f'<span class="badge {cls}">{txt}</span>'


def _ch_tab_panel(group_id: str, available: list, figs: dict, no_data_msg: str) -> str:
    """채널 전환 탭 + 패널 HTML 생성 헬퍼."""
    first = available[0] if available else ""
    tabs_html = "".join(
        f'<div class="ch-tab {"active" if ch == first else ""}" '
        f"data-ch=\"{ch}\" onclick=\"switchChTab('{group_id}','{ch}')\">{ch}</div>"
        for ch in available
    )
    panels_html = ""
    for ch in available:
        active = "active" if ch == first else ""
        content = figs.get(ch) or f'<div class="no-data">{no_data_msg} [{ch}]</div>'
        panels_html += (
            f'<div class="ch-panel {active}" id="{group_id}-panel-{ch}">{content}</div>'
        )
    return f'<div class="ch-tabs" id="{group_id}">{tabs_html}</div>{panels_html}'


def _build_sec_summary(metrics: dict, available: list) -> str:
    overall = metrics.get("overall", {})
    avg_acc = overall.get("accuracy", 0.0)
    avg_f1 = overall.get("macro_f1", 0.0)
    avg_mae = overall.get("mae", 0.0)

    acc_cls = "pass-card" if avg_acc >= TARGET_OVERALL_ACC else "fail-card"
    f1_cls = "pass-card" if avg_f1 >= TARGET_PER_CLASS_F1 else "fail-card"
    mae_cls = "pass-card" if avg_mae <= TARGET_MAE else "fail-card"

    kpi = f"""
    <div class="kpi-grid">
      <div class="kpi-card {acc_cls}">
        <div class="lbl">Overall Accuracy</div>
        <div class="val">{avg_acc:.1%}</div>
        <div class="tgt">Target >= {TARGET_OVERALL_ACC:.0%}</div>
      </div>
      <div class="kpi-card {f1_cls}">
        <div class="lbl">Macro F1</div>
        <div class="val">{avg_f1:.4f}</div>
        <div class="tgt">Target >= {TARGET_PER_CLASS_F1:.2f}</div>
      </div>
      <div class="kpi-card {mae_cls}">
        <div class="lbl">Overall MAE</div>
        <div class="val">{avg_mae:.4f}</div>
        <div class="tgt">Target <= {TARGET_MAE:.2f}</div>
      </div>
    </div>"""

    rows = ""
    for ch in available:
        m = metrics.get(ch, {})
        acc = m.get("accuracy", 0.0)
        f1 = m.get("macro_f1", 0.0)
        mae = m.get("mae", 0.0)
        pass_acc = acc >= TARGET_PER_COLOR_ACC
        pass_mae = mae <= TARGET_MAE
        pass_f1 = f1 >= TARGET_PER_CLASS_F1
        rows += f"""
        <tr>
          <td><span class="ch-badge ch-{ch}">{ch}</span></td>
          <td>{acc:.4f} {_badge(pass_acc)}</td>
          <td>{f1:.4f} {_badge(pass_f1)}</td>
          <td>{mae:.4f} {_badge(pass_mae)}</td>
        </tr>"""

    table = f"""
    <div class="card">
      <h2>Channel Evaluation Results (best_{{ch}}.pt — Test-set)</h2>
      <table>
        <thead><tr>
          <th>Channel</th>
          <th>Accuracy (>={TARGET_PER_COLOR_ACC:.0%})</th>
          <th>Macro F1 (>={TARGET_PER_CLASS_F1:.2f})</th>
          <th>MAE (&lt;={TARGET_MAE:.2f})</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return kpi + table


def _build_sec_params(params_by_channel: dict, cfg: dict, available: list) -> str:
    """
    Optuna Best Params 탭.
    기본값 대비 변화한 파라미터는 주황색으로 강조 표시한다.
    """
    # config.json 기본값 수집
    p2 = cfg.get("phase2", {})
    backbone = cfg.get("model", {}).get("backbone", "efficientnet_b0")
    head_def = cfg.get("phase2", {}).get("heads", {}).get(backbone, {})
    defaults = {
        "learning_rate": p2.get("learning_rate"),
        "weight_decay": p2.get("weight_decay"),
        "batch_size": p2.get("batch_size"),
        "epochs": p2.get("epochs"),
        "dropout": head_def.get("dropout"),
        "hidden_dim": head_def.get("hidden_dim"),
        "mid_dim": head_def.get("mid_dim"),
    }

    all_keys = [
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "dropout",
        "hidden_dim",
        "mid_dim",
    ]

    # 헤더행
    header = "<tr><th>Parameter</th><th>Default</th>"
    for ch in available:
        header += f"<th><span class='ch-badge ch-{ch}'>{ch}</span></th>"
    header += "</tr>"

    rows = ""
    for key in all_keys:
        def_val = defaults.get(key)
        cells = f"<td><code>{key}</code></td><td class='same'>{def_val if def_val is not None else '—'}</td>"
        for ch in available:
            params = params_by_channel.get(ch)
            if params is None or key not in params:
                cells += "<td class='same'>—</td>"
                continue
            val = params[key]
            changed = def_val is not None and val != def_val
            cls = "changed" if changed else "same"
            # 과학적 표기법은 float인 경우만
            if isinstance(val, float):
                fmt = f"{val:.2e}"
            else:
                fmt = str(val)
            cells += f"<td class='{cls}'>{fmt}</td>"
        rows += f"<tr>{cells}</tr>"

    table = f"""
    <div class="card">
      <h2>Best Hyperparameters (채널별 Optuna 최적값 / 주황색 = 기본값과 다름)</h2>
      <p style="color:var(--fm);font-size:.82rem;margin-bottom:1rem">
        Backbone: <code>{backbone}</code> &nbsp;|&nbsp;
        Default epochs: {defaults.get('epochs')} &nbsp;|&nbsp;
        Default lr: {defaults.get('learning_rate')}
      </p>
      <table>
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    # Best val_acc 행 추가 (trials_summary에서 추출)
    best_vals = ""
    for ch in available:
        params = params_by_channel.get(ch)
        if params and "_best_val_acc" in params:
            best_vals += f"<li><span class='ch-badge ch-{ch}'>{ch}</span> best val_acc = {params['_best_val_acc']:.4f}</li>"

    return table


def _build_sec_trials(
    trial_figs: dict, trials_by_channel: dict, available: list
) -> str:
    """Trial History 탭 — 최적화 곡선 + 완료/Pruned 통계."""
    stats_rows = ""
    for ch in available:
        trials = trials_by_channel.get(ch)
        if not trials:
            stats_rows += f"<tr><td><span class='ch-badge ch-{ch}'>{ch}</span></td><td colspan='4' class='same'>데이터 없음</td></tr>"
            continue
        total = len(trials)
        complete = sum(1 for t in trials if t.get("state") == "TrialState.COMPLETE")
        pruned = sum(1 for t in trials if t.get("state") == "TrialState.PRUNED")
        failed = total - complete - pruned
        best_val = max(
            (
                t["value"]
                for t in trials
                if t.get("state") == "TrialState.COMPLETE"
                and t.get("value") is not None
            ),
            default=None,
        )
        best_str = f"{best_val:.4f}" if best_val is not None else "—"
        stats_rows += f"""
        <tr>
          <td><span class='ch-badge ch-{ch}'>{ch}</span></td>
          <td>{total}</td><td>{complete}</td><td>{pruned}</td><td>{failed}</td>
          <td>{best_str}</td>
        </tr>"""

    stats_table = f"""
    <div class="card">
      <h2>Trial Statistics</h2>
      <table>
        <thead><tr>
          <th>Channel</th><th>Total</th><th>Complete</th>
          <th>Pruned</th><th>Failed</th><th>Best Val Acc</th>
        </tr></thead>
        <tbody>{stats_rows}</tbody>
      </table>
    </div>"""

    curves_html = _ch_tab_panel("trials", available, trial_figs, "Trial history 없음")
    curves_card = f"""
    <div class="card">
      <h2>Optimization Curve (Trial # vs. Val Accuracy)</h2>
      {curves_html}
    </div>"""

    return stats_table + curves_card


def _build_sec_cm(cm_figs: dict, available: list) -> str:
    per_ch = "".join(cm_figs.get(f"cm_{ch}", "") for ch in available)
    overall = cm_figs.get("cm_overall", "")
    return f"""
    <div class="card">
      <h2>Confusion Matrix (Row-Normalized) — Per Channel</h2>
      <div class="cmg">{per_ch if per_ch else '<div class="no-data">추론 결과 없음</div>'}</div>
    </div>
    <div class="card">
      <h2>Confusion Matrix — Overall</h2>
      {overall if overall else '<div class="no-data">Overall CM 없음</div>'}
    </div>"""


def _build_sec_f1mae(
    f1_figs: dict, mae_heat: str, metrics: dict, available: list
) -> str:
    f1_panels = _ch_tab_panel("f1tabs", available, f1_figs, "Per-class 데이터 없음")
    f1_card = f"""
    <div class="card">
      <h2>Per-Class F1 / Precision / Recall</h2>
      {f1_panels}
    </div>"""

    # Overall per-level table
    pc_rows = ""
    if "overall" in metrics:
        for pc in metrics["overall"].get("per_class", []):
            pass_f1 = pc["f1"] >= TARGET_PER_CLASS_F1
            pc_rows += f"""
            <tr>
              <td>Level {pc['level']}</td>
              <td>{pc['precision']:.4f}</td>
              <td>{pc['recall']:.4f}</td>
              <td>{pc['f1']:.4f} {_badge(pass_f1)}</td>
              <td>{pc.get('mae', 0.0):.4f}</td>
            </tr>"""

    table_card = ""
    if pc_rows:
        table_card = f"""
    <div class="card">
      <h2>Overall Per-Level Summary (F1 target >= {TARGET_PER_CLASS_F1:.2f})</h2>
      <table>
        <thead><tr>
          <th>Level</th><th>Precision</th><th>Recall</th>
          <th>F1</th><th>MAE</th>
        </tr></thead>
        <tbody>{pc_rows}</tbody>
      </table>
    </div>"""

    mae_card = f"""
    <div class="card">
      <h2>MAE Heatmap (Channel x True Level — target <= {TARGET_MAE})</h2>
      {mae_heat if mae_heat else '<div class="no-data">추론 결과 없음</div>'}
    </div>"""

    return f1_card + table_card + mae_card


# ── 통합 HTML 빌더 ────────────────────────────────────────────────────────────


def build_optuna_html(report_data: dict, output_path: Path) -> None:
    """6탭 Optuna 통합 HTML 리포트를 생성한다."""
    meta = report_data["meta"]
    metrics = report_data["metrics"]
    figs = report_data["figures"]
    params_by_channel = report_data["params_by_channel"]
    trials_by_channel = report_data["trials_by_channel"]
    available = report_data["available"]
    cfg = report_data["cfg"]

    sec_summary = _build_sec_summary(metrics, available)
    sec_params = _build_sec_params(params_by_channel, cfg, available)
    sec_trials = _build_sec_trials(
        figs.get("trial_history", {}), trials_by_channel, available
    )
    sec_cm = _build_sec_cm(figs.get("confusion", {}), available)
    sec_f1mae = _build_sec_f1mae(
        figs.get("per_class", {}),
        figs.get("mae_heatmap", ""),
        metrics,
        available,
    )
    sec_conf = f"""
    <div class="card">
      <h2>Prediction Confidence Distribution</h2>
      {figs.get('conf_dist') or '<div class="no-data">추론 결과 없음</div>'}
    </div>"""

    tabs = [
        ("summary", "① Summary"),
        ("params", "② Optuna Params"),
        ("trials", "③ Trial History"),
        ("cm", "④ Confusion Matrix"),
        ("f1mae", "⑤ Per-Class F1 / MAE"),
        ("conf", "⑥ Confidence"),
    ]
    tabs_html = "".join(
        f'<div class="nav-tab {"active" if i == 0 else ""}" '
        f'data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</div>'
        for i, (tid, label) in enumerate(tabs)
    )

    sections_map = {
        "summary": sec_summary,
        "params": sec_params,
        "trials": sec_trials,
        "cm": sec_cm,
        "f1mae": sec_f1mae,
        "conf": sec_conf,
    }
    sections_html = "".join(
        f'<section id="sec-{tid}" class="section {"active" if i == 0 else ""}">'
        f"{sections_map[tid]}</section>"
        for i, (tid, _) in enumerate(tabs)
    )

    generated_at = meta["generated_at"]
    backbone = meta["backbone"]
    channels_str = ", ".join(available)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Optuna Phase 2 Report</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body>

<div class="rh">
  <h1>Grayspot — Optuna Phase 2 Report</h1>
  <div class="meta">
    Generated: {generated_at} &nbsp;|&nbsp;
    Backbone: {backbone} &nbsp;|&nbsp;
    Channels: {channels_str}
  </div>
</div>

<nav class="nav-tabs">{tabs_html}</nav>
<main>{sections_html}</main>

<div class="rf">
  Grayspot Detection Pipeline &middot; Optuna Phase 2 Report &middot; {generated_at}
</div>

<script>{_TAB_JS}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Optuna Phase 2 unified HTML report"
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to include (default: all with best_params file)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated HTML in the default browser",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip model loading; show params/trial history only",
    )
    args = parser.parse_args()

    import torch

    cfg = load_config()
    device = torch.device(cfg["system"]["device"])

    # ── 1. 사용 가능한 채널 결정 ───────────────────────────────────────────
    if args.channels:
        available = [
            c.upper()
            for c in args.channels
            if (OPTUNA_DIR / f"best_params_phase2_{c.lower()}.json").exists()
        ]
    else:
        available = [
            c
            for c in _ALL_CHANNELS
            if (OPTUNA_DIR / f"best_params_phase2_{c.lower()}.json").exists()
        ]

    if not available:
        print(
            "[ERROR] best_params_phase2_*.json 파일이 없습니다.\n"
            f"        Run: python -m src.scripts.run_optuna --phase 2\n"
            f"        Searched: {OPTUNA_DIR}"
        )
        sys.exit(1)

    print(f"[Report] Channels: {available}")

    # ── 2. Optuna 결과 로드 ────────────────────────────────────────────────
    params_by_channel = {ch: load_best_params(ch) for ch in available}
    trials_by_channel = {ch: load_trials_summary(ch) for ch in available}

    for ch in available:
        p = params_by_channel.get(ch)
        print(f"  [{ch}] best_params: {p}")

    # ── 3. 추론 + 지표 계산 ────────────────────────────────────────────────
    all_results: dict = {}
    metrics: dict = {}

    if not args.no_inference:
        labeled_dir = Path(cfg["storage"]["labeled_dir"])
        labels_csv = Path(cfg["storage"]["data_root"]) / "labels_master.csv"
        tmp_dir = REPORT_DIR / "tmp_optuna"

        for ch in available:
            ckpt = find_best_checkpoint(ch)
            if ckpt is None:
                print(f"  [{ch}] best_{ch}.pt 없음 — 추론 skip")
                continue
            print(f"  [{ch}] Inference from: {ckpt}")
            try:
                model = build_model(cfg, ckpt, device)
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
                print(f"  [{ch}] Done")
            except Exception as exc:
                print(f"  [{ch}] [WARN] Inference failed: {exc}")

        if all_results:
            metrics = compute_all_channels(
                all_results,
                list(all_results.keys()),
                num_classes=cfg["data"]["num_levels"],
            )
    else:
        print("[Report] --no-inference: 추론 생략")

    # ── 4. Figures 빌드 ───────────────────────────────────────────────────
    import plotly.io as pio

    figs: dict = {}

    figs["trial_history"] = _build_trial_history_figs(trials_by_channel, available)

    if all_results:
        figs["per_class"] = _build_per_class_bars(metrics, list(all_results.keys()))
        figs["mae_heatmap"] = _build_mae_heatmap(all_results, list(all_results.keys()))
        figs["conf_dist"] = _build_conf_dist(all_results, list(all_results.keys()))

        cm_figs: dict = {}
        for ch in all_results:
            yt, yp = all_results[ch]["y_true"], all_results[ch]["y_pred"]
            ch_acc = metrics.get(ch, {}).get("accuracy", 0.0)
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
    else:
        figs["per_class"] = {ch: None for ch in available}
        figs["mae_heatmap"] = ""
        figs["conf_dist"] = ""
        figs["confusion"] = {}

    # ── 5. HTML 출력 ──────────────────────────────────────────────────────
    output_path = REPORT_DIR / "optuna_phase2_report.html"
    build_optuna_html(
        report_data={
            "meta": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "backbone": cfg["model"]["backbone"],
                "channels": available,
            },
            "metrics": metrics,
            "figures": figs,
            "params_by_channel": params_by_channel,
            "trials_by_channel": trials_by_channel,
            "available": available,
            "cfg": cfg,
        },
        output_path=output_path,
    )

    print(f"\n[Report] Saved: {output_path}")

    if args.open_browser:
        import webbrowser

        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
