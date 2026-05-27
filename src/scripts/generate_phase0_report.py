"""
scripts/generate_phase0_report.py

Phase 0 Contrastive Learning 결과 시각화 리포트 생성 스크립트.
Phase 0 Contrastive Learning result visualization report generator.

Phase 0 는 비지도 SimCLR 학습이므로 accuracy 지표 없이
loss 수렴 곡선 + t-SNE embedding 시각화로 학습 품질을 평가한다.
Since Phase 0 is unsupervised SimCLR, learning quality is assessed via
loss convergence curves + t-SNE embedding visualization (no accuracy metric).

입력 / Inputs:
    outputs/checkpoints/phase0_summary.json       — 채널별 final loss 요약
    outputs/checkpoints/phase0_history_{ch}.csv   — epoch별 loss / lr / elapsed
    data_set/models/phase0_backbone_{ch}_{tag}.pt — t-SNE용 backbone 체크포인트

출력 / Outputs:
    outputs/reports/phase0.html   — 통합 HTML 리포트

실행 / Run:
    python -m src.scripts.generate_phase0_report
    python -m src.scripts.generate_phase0_report --no-tsne
    python -m src.scripts.generate_phase0_report --open-browser

전제 조건 / Prerequisites:
    - run_phase0.py 실행 완료
      -> outputs/checkpoints/phase0_summary.json
      -> outputs/checkpoints/phase0_history_{ch}.csv

Python 3.11.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils import (
    create_directories,
    get_logger,
    get_nested,
    load_config,
    setup_logging,
    validate_config,
)

CKPT_DIR = ROOT / "outputs" / "checkpoints"
REPORT_DIR = ROOT / "outputs" / "reports"

logger = get_logger(__name__)

# ── 상수 / Constants ──────────────────────────────────────────────────────────

_FEATURE_TO_BACKBONE: dict = {
    2048: "resnet50",
    1280: "efficientnet_b0",
    1792: "efficientnet_b4",
    1536: "efficientnet_b3",
    1408: "efficientnet_b2",
    512: "resnet18",
    1024: "densenet121",
}

# ── CSS / JS ─────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0b1220; color: #e6eef8; font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 14px; }
header { background: #111c30; padding: 18px 32px; border-bottom: 2px solid #1e3a5f; display:flex; align-items:center; gap:16px; }
header h1 { font-size: 1.35rem; font-weight: 700; color: #7eb8f7; }
header .meta { font-size: 0.78rem; color: #7a9cbf; }
.nav { background: #0d1929; padding: 0 24px; display:flex; gap:4px; border-bottom: 1px solid #1e3a5f; }
.nav-tab { padding: 10px 18px; cursor: pointer; border-bottom: 3px solid transparent; color: #8aadcc; font-size: 0.85rem; transition: all .15s; }
.nav-tab:hover { color: #c8dff0; }
.nav-tab.active { color: #7eb8f7; border-bottom-color: #3b82f6; font-weight: 600; }
.content { padding: 24px 32px; }
.section { display: none; }
.section.active { display: block; }
.card { background: #111c30; border: 1px solid #1e3a5f; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; }
.card h2 { font-size: 1rem; font-weight: 600; color: #7eb8f7; margin-bottom: 14px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 14px; margin-bottom: 20px; }
.kpi-card { background: #0d1929; border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; text-align: center; }
.kpi-card .lbl { font-size: 0.75rem; color: #7a9cbf; margin-bottom: 6px; }
.kpi-card .val { font-size: 1.6rem; font-weight: 700; color: #7eb8f7; }
.kpi-card .sub { font-size: 0.72rem; color: #566a80; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th { background: #0d1929; color: #7a9cbf; padding: 8px 12px; text-align: left; border-bottom: 1px solid #1e3a5f; }
td { padding: 7px 12px; border-bottom: 1px solid #162035; }
tr:hover td { background: #0f1f33; }
.ch-badge { display:inline-block; padding: 2px 8px; border-radius: 4px; font-weight:700; font-size:0.8rem; }
.ch-Y { background:#4a3a00; color:#f5e642; }
.ch-M { background:#3a0020; color:#e91e8c; }
.ch-C { background:#003040; color:#00b4d8; }
.ch-K { background:#202020; color:#aaa; }
.ch-tabs { display:flex; gap:6px; margin-bottom:14px; flex-wrap:wrap; }
.ch-tab { padding: 6px 14px; border-radius: 4px; cursor:pointer; background:#0d1929; color:#8aadcc; font-size:0.82rem; border: 1px solid #1e3a5f; }
.ch-tab.active { background:#1e3a5f; color:#7eb8f7; font-weight:600; }
.ch-panel { display:none; }
.ch-panel.active { display:block; }
.no-data { padding: 28px; text-align: center; color: #4a6075; border: 1px dashed #1e3a5f; border-radius: 6px; }
.conv-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap:14px; }
.conv-item { background:#0d1929; border:1px solid #1e3a5f; border-radius:6px; padding:14px; }
.conv-item .ch-title { font-weight:700; margin-bottom:8px; }
.conv-item .stat { display:flex; justify-content:space-between; font-size:0.82rem; color:#7a9cbf; padding:3px 0; }
.conv-item .stat span:last-child { color:#e6eef8; }
"""

_JS = """
function switchTab(tid) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelector('.nav-tab[data-tab="' + tid + '"]').classList.add('active');
    document.getElementById('sec-' + tid).classList.add('active');
}
function switchChTab(groupId, ch) {
    document.querySelectorAll('#' + groupId + ' .ch-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('[id^="' + groupId + '-panel-"]').forEach(p => p.classList.remove('active'));
    var tabEl = document.querySelector('#' + groupId + ' .ch-tab[data-ch="' + ch + '"]');
    var panEl = document.getElementById(groupId + '-panel-' + ch);
    if (tabEl) tabEl.classList.add('active');
    if (panEl) panEl.classList.add('active');
}
"""


# ── 데이터 로더 / Data loaders ────────────────────────────────────────────────


def load_phase0_summary() -> dict:
    path = CKPT_DIR / "phase0_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"phase0_summary.json not found: {path}\n" "Run run_phase0.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_history(ch: str) -> list[dict]:
    path = CKPT_DIR / f"phase0_history_{ch}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {k: float(v) if k != "epoch" else int(float(v)) for k, v in row.items()}
            )
    return rows


def find_backbone(ch: str, backbone_name: str) -> Path | None:
    from src.utils.utils_model import backbone_tag

    tag = backbone_tag(backbone_name)
    candidates = [
        CKPT_DIR.parent.parent
        / "data_set"
        / "models"
        / f"phase0_backbone_{ch}_{tag}.pt",
        ROOT / "data_set" / "models" / f"phase0_backbone_{ch}_{tag}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── 차트 빌더 / Figure builders ───────────────────────────────────────────────


def _build_loss_curves(channels: list[str]) -> dict[str, str]:
    """채널별 loss + lr 곡선 / Per-channel loss + LR curves."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    figs = {}
    for ch in channels:
        history = load_history(ch)
        if not history:
            figs[ch] = None
            continue

        epochs = [r["epoch"] for r in history]
        losses = [r["loss"] for r in history]
        lrs = [r.get("lr", 0) for r in history]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"[{ch}] InfoNCE Loss", f"[{ch}] Learning Rate"],
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses,
                name="Loss",
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=lrs,
                name="LR",
                mode="lines",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(11,18,32,0.6)",
            paper_bgcolor="rgba(11,18,32,0)",
            height=340,
            margin=dict(l=40, r=20, t=40, b=30),
            showlegend=False,
        )
        figs[ch] = pio.to_html(
            fig, full_html=False, include_plotlyjs=False, div_id=f"fig-loss-{ch}"
        )
    return figs


def _build_all_channels_loss(channels: list[str]) -> str:
    """전체 채널 loss 비교 차트 / All-channel loss comparison."""
    import plotly.graph_objects as go
    import plotly.io as pio

    CH_COLORS = {"Y": "#f5e642", "M": "#e91e8c", "C": "#00b4d8", "K": "#aaaaaa"}
    fig = go.Figure()
    for ch in channels:
        history = load_history(ch)
        if not history:
            continue
        epochs = [r["epoch"] for r in history]
        losses = [r["loss"] for r in history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=losses,
                name=f"Channel {ch}",
                mode="lines+markers",
                line=dict(color=CH_COLORS.get(ch, "#888"), width=2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(11,18,32,0.6)",
        paper_bgcolor="rgba(11,18,32,0)",
        xaxis_title="Epoch",
        yaxis_title="InfoNCE Loss",
        height=380,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False, div_id="fig-all-loss"
    )


def _build_tsne(channels: list[str], cfg: dict, device: torch.device) -> str:
    """
    Phase 0 backbone으로 t-SNE 2D 임베딩 시각화를 생성한다.
    Generates t-SNE 2D embedding visualization using Phase 0 backbone.

    각 채널 backbone의 임베딩을 레벨별로 색상 구분하여 플롯한다.
    Plots embeddings from each channel backbone colored by level.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return "<div class='no-data'>scikit-learn required for t-SNE. Install: pip install scikit-learn</div>"

    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    from torch.utils.data import DataLoader

    from data.dataset import CMYKDataset

    backbone_name = cfg["model"]["backbone"]
    image_size = cfg["data"]["image_size"]
    LEVEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    all_embeds: list[np.ndarray] = []
    all_levels: list[int] = []
    all_chs: list[str] = []

    for ch in channels:
        ckpt_path = find_backbone(ch, backbone_name)
        if ckpt_path is None:
            logger.warning(
                f"[{ch}] backbone not found — skipping t-SNE for this channel"
            )
            continue
        try:
            import copy

            from models.grayspot_model import GrayspotModel

            ch_cfg = copy.deepcopy(cfg)
            model = GrayspotModel(ch_cfg, phase=0)
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model = model.to(device).eval()

            ds = CMYKDataset(cfg, channel=ch)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

            embeds, levels_list = [], []
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = imgs.to(device)
                    feats = model.backbone(imgs)
                    if feats.dim() > 2:
                        feats = feats.mean(dim=[2, 3])
                    embeds.append(feats.cpu().numpy())
                    levels_list.extend(lbls.numpy().tolist())

            if embeds:
                all_embeds.append(np.concatenate(embeds, axis=0))
                all_levels.extend(levels_list)
                all_chs.extend([ch] * len(levels_list))
                logger.info(f"[{ch}] embeddings extracted: {len(levels_list)} samples")
        except Exception as e:
            logger.warning(f"[{ch}] embedding extraction failed: {e}")

    if not all_embeds:
        return "<div class='no-data'>No embeddings available for t-SNE.</div>"

    X = np.concatenate(all_embeds, axis=0)
    logger.info(f"Running t-SNE on {X.shape[0]} samples × {X.shape[1]} dims ...")

    # 샘플 너무 많으면 랜덤 서브샘플링 (최대 3000)
    MAX_SAMPLES = 3000
    if len(X) > MAX_SAMPLES:
        idx = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[idx]
        all_levels = [all_levels[i] for i in idx]
        all_chs = [all_chs[i] for i in idx]

    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    X2d = tsne.fit_transform(X)

    # ── 서브플롯: 레벨별 색상 + 채널별 색상 ──────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Colored by Grayspot Level", "Colored by Channel"],
    )

    # 레벨별
    for lv in range(cfg["data"]["num_levels"]):
        mask = [i for i, l in enumerate(all_levels) if l == lv]
        if not mask:
            continue
        fig.add_trace(
            go.Scatter(
                x=X2d[mask, 0],
                y=X2d[mask, 1],
                mode="markers",
                name=f"Level {lv}",
                marker=dict(size=4, color=LEVEL_COLORS[lv], opacity=0.7),
            ),
            row=1,
            col=1,
        )

    # 채널별
    CH_COLORS = {"Y": "#f5e642", "M": "#e91e8c", "C": "#00b4d8", "K": "#aaaaaa"}
    for ch in set(all_chs):
        mask = [i for i, c in enumerate(all_chs) if c == ch]
        fig.add_trace(
            go.Scatter(
                x=X2d[mask, 0],
                y=X2d[mask, 1],
                mode="markers",
                name=f"Ch {ch}",
                marker=dict(size=4, color=CH_COLORS.get(ch, "#888"), opacity=0.7),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(11,18,32,0.6)",
        paper_bgcolor="rgba(11,18,32,0)",
        height=500,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id="fig-tsne")


# ── HTML 빌더 / HTML builder ──────────────────────────────────────────────────


def build_phase0_html(report_data: dict, output_path: Path) -> None:
    """Phase 0 통합 HTML 리포트를 생성한다."""
    meta = report_data["meta"]
    summary = report_data["summary"]
    available = report_data["available"]
    figs = report_data["figures"]

    generated_at = meta["generated_at"]
    backbone = meta["backbone"]
    epochs = meta["epochs"]

    # ── Tab 1: Summary ──────────────────────────────────────────────────────
    rows = ""
    total_time = 0.0
    for r in summary["results"]:
        if r.get("skipped"):
            continue
        ch = r["channel"]
        history = load_history(ch)
        elapsed = sum(row.get("elapsed", 0) for row in history)
        total_time += elapsed
        init_loss = history[0]["loss"] if history else float("nan")
        final_loss = (
            history[-1]["loss"] if history else r.get("final_loss", float("nan"))
        )
        improvement = init_loss - final_loss if history else 0.0
        rows += f"""
        <tr>
          <td><span class="ch-badge ch-{ch}">{ch}</span></td>
          <td>{r.get('n_images', '—')}</td>
          <td>{r.get('epochs', epochs)}</td>
          <td>{init_loss:.4f}</td>
          <td>{final_loss:.4f}</td>
          <td>{improvement:.4f}</td>
          <td>{elapsed:.0f}s</td>
        </tr>"""

    n_channels = len([r for r in summary["results"] if not r.get("skipped")])
    avg_loss = sum(
        r["final_loss"] for r in summary["results"] if not r.get("skipped")
    ) / max(n_channels, 1)

    sec_summary = f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="lbl">Channels Trained</div>
        <div class="val">{n_channels}</div><div class="sub">of 4 CMYK</div></div>
      <div class="kpi-card"><div class="lbl">Avg. Final Loss</div>
        <div class="val">{avg_loss:.4f}</div><div class="sub">InfoNCE</div></div>
      <div class="kpi-card"><div class="lbl">Epochs</div>
        <div class="val">{epochs}</div><div class="sub">per channel</div></div>
      <div class="kpi-card"><div class="lbl">Total Training Time</div>
        <div class="val">{total_time/60:.1f}m</div><div class="sub">all channels</div></div>
    </div>
    <div class="card">
      <h2>Channel Training Summary</h2>
      <table>
        <thead><tr>
          <th>Channel</th><th>Images</th><th>Epochs</th>
          <th>Initial Loss</th><th>Final Loss</th><th>Improvement</th><th>Time</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    # ── Tab 2: Loss Curves ──────────────────────────────────────────────────
    all_ch_loss = figs.get("all_channels_loss", "")
    curve_figs = figs.get("loss_curves", {})
    first_ch = available[0] if available else "Y"

    ch_tabs = "".join(
        f'<div class="ch-tab{"active" if ch == first_ch else ""}" '
        f"data-ch=\"{ch}\" onclick=\"switchChTab('loss','{ch}')\">{ch}</div>"
        for ch in available
    )
    ch_panels = ""
    for ch in available:
        active = "active" if ch == first_ch else ""
        content = (
            curve_figs.get(ch)
            or f'<div class="no-data">No history data for {ch}.</div>'
        )
        ch_panels += (
            f'<div class="ch-panel {active}" id="loss-panel-{ch}">{content}</div>'
        )

    sec_curves = f"""
    <div class="card">
      <h2>All Channels — InfoNCE Loss Comparison</h2>
      {all_ch_loss if all_ch_loss else '<div class="no-data">No data available.</div>'}
    </div>
    <div class="card">
      <h2>Per-Channel Loss + LR Schedule</h2>
      <div class="ch-tabs" id="loss">{ch_tabs}</div>
      {ch_panels}
    </div>"""

    # ── Tab 3: t-SNE Embeddings ─────────────────────────────────────────────
    tsne_div = figs.get("tsne", "")
    sec_tsne = f"""
    <div class="card">
      <h2>t-SNE Embedding Visualization (Phase 0 Backbone)</h2>
      <p style="color:#7a9cbf;font-size:0.82rem;margin-bottom:14px;">
        Phase 0 backbone으로 추출한 임베딩을 t-SNE로 2D 투영합니다.
        왼쪽: Grayspot 레벨별 색상 / 오른쪽: CMYK 채널별 색상.<br>
        클러스터가 레벨별로 분리될수록 backbone의 표현 품질이 높습니다.
      </p>
      {tsne_div if tsne_div else '<div class="no-data">t-SNE not available (use --no-tsne to skip, or run without flag to generate).</div>'}
    </div>"""

    # ── Tab 4: Convergence Analysis ─────────────────────────────────────────
    conv_items = ""
    for ch in available:
        history = load_history(ch)
        if not history:
            continue
        losses = [r["loss"] for r in history]
        best_epoch = int(np.argmin(losses)) + 1
        converge_rate = (losses[0] - losses[-1]) / max(losses[0], 1e-8) * 100
        plateau = losses[-1] / max(losses[0], 1e-8)
        conv_items += f"""
        <div class="conv-item">
          <div class="ch-title"><span class="ch-badge ch-{ch}">{ch}</span></div>
          <div class="stat"><span>Initial Loss</span><span>{losses[0]:.4f}</span></div>
          <div class="stat"><span>Final Loss</span><span>{losses[-1]:.4f}</span></div>
          <div class="stat"><span>Best Loss Epoch</span><span>{best_epoch}</span></div>
          <div class="stat"><span>Loss Reduction</span><span>{converge_rate:.1f}%</span></div>
          <div class="stat"><span>Final / Initial Ratio</span><span>{plateau:.3f}</span></div>
          <div class="stat"><span>Total Epochs</span><span>{len(history)}</span></div>
        </div>"""

    sec_conv = f"""
    <div class="card">
      <h2>Convergence Analysis — 수렴 분석</h2>
      <p style="color:#7a9cbf;font-size:0.82rem;margin-bottom:14px;">
        Loss Reduction: 초기 대비 최종 loss 감소율. 높을수록 수렴 양호.<br>
        Final/Initial Ratio: 1에 가까울수록 수렴 불량.
      </p>
      <div class="conv-grid">{conv_items}</div>
    </div>"""

    # ── 탭 조립 / Assemble tabs ─────────────────────────────────────────────
    tabs = [
        ("summary", "① Summary"),
        ("curves", "② Loss Curves"),
        ("tsne", "③ t-SNE Embeddings"),
        ("conv", "④ Convergence"),
    ]
    tabs_html = "".join(
        f'<div class="nav-tab{"active" if i == 0 else ""}" '
        f'data-tab="{tid}" onclick="switchTab(\'{tid}\')">{label}</div>'
        for i, (tid, label) in enumerate(tabs)
    )
    sections_map = {
        "summary": sec_summary,
        "curves": sec_curves,
        "tsne": sec_tsne,
        "conv": sec_conv,
    }
    sections_html = "".join(
        f'<section id="sec-{tid}" class="section{"active" if i == 0 else ""}">'
        f"{sections_map[tid]}</section>"
        for i, (tid, _) in enumerate(tabs)
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 0 Report — Grayspot</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>Phase 0 — Contrastive Learning Report</h1>
    <div class="meta">
      Backbone: {backbone} &nbsp;|&nbsp;
      Epochs: {epochs} &nbsp;|&nbsp;
      Generated: {generated_at}
    </div>
  </div>
</header>
<nav class="nav">{tabs_html}</nav>
<div class="content">{sections_html}</div>
<script>{_JS}</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved / 저장: {output_path}")


# ── 메인 / Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 Contrastive Learning 결과 시각화 리포트 생성"
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        help="t-SNE 임베딩 시각화 생략 (빠른 실행 / skip for speed)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="처리할 채널 (기본: summary에서 자동 감지)",
    )
    parser.add_argument(
        "--open-browser", action="store_true", help="완료 후 브라우저 자동 열기"
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

    logger.info("=" * 60)
    logger.info("Phase 0 Report Generation")
    logger.info("=" * 60)

    # ── 1. Summary 로드 ───────────────────────────────────────────────────
    summary = load_phase0_summary()
    backbone = summary.get("backbone", cfg["model"]["backbone"])
    epochs = summary.get(
        "epochs",
        (
            summary["results"][0]["epochs"]
            if summary.get("results")
            else cfg["phase0"]["epochs"]
        ),
    )

    available = args.channels or [
        r["channel"] for r in summary.get("results", []) if not r.get("skipped")
    ]
    logger.info(f"Channels: {available}")
    logger.info(f"Backbone: {backbone} | Epochs: {epochs}")

    device = torch.device(cfg["system"]["device"])

    # ── 2. 차트 생성 ──────────────────────────────────────────────────────
    logger.info("Building figures...")
    figs: dict = {}

    figs["all_channels_loss"] = _build_all_channels_loss(available)
    figs["loss_curves"] = _build_loss_curves(available)

    if args.no_tsne:
        logger.info("--no-tsne: skipping t-SNE")
        figs["tsne"] = ""
    else:
        logger.info("Extracting embeddings for t-SNE (may take a while)...")
        figs["tsne"] = _build_tsne(available, cfg, device)

    # ── 3. HTML 생성 ──────────────────────────────────────────────────────
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": backbone,
        "epochs": epochs,
        "channels": available,
    }

    output_path = REPORT_DIR / "phase0.html"
    build_phase0_html(
        report_data={
            "meta": meta,
            "summary": summary,
            "available": available,
            "figures": figs,
        },
        output_path=output_path,
    )

    logger.info("=" * 60)
    logger.info(f"Report complete -> {output_path}")
    logger.info("=" * 60)
    print(f"\nReport saved: {output_path}")

    if args.open_browser:
        import webbrowser

        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
