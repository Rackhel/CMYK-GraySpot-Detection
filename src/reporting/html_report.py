"""
Grayspot — HTML 평가 리포트 생성
reporting/html_report.py

Phase 3 평가 결과를 Grayspot PRD 스타일의 HTML 리포트로 저장한다.
포함 내용: 채널별 메트릭, Confusion Matrix, Per-class F1, Swing 판단 결과
"""

import json
from pathlib import Path
from datetime import datetime

CHANNELS = ["Y", "M", "C", "K"]
CHANNEL_COLORS = {
    "Y": ("#ffeb3b", "#f9a825"),   # Yellow
    "M": ("#f48fb1", "#c2185b"),   # Magenta
    "C": ("#81d4fa", "#0277bd"),   # Cyan
    "K": ("#bdbdbd", "#424242"),   # Black
}
SWING_BADGE = {
    "pass":   ('<span style="background:rgba(80,227,194,.15);color:#50e3c2;border:1px solid rgba(80,227,194,.3);padding:2px 10px;border-radius:10px;font-size:.78rem;font-weight:700;"> PASS</span>'),
    "phase1": ('<span style="background:rgba(255,179,71,.15);color:#ffb347;border:1px solid rgba(255,179,71,.3);padding:2px 10px;border-radius:10px;font-size:.78rem;font-weight:700;"> → Phase 1</span>'),
    "phase0": ('<span style="background:rgba(255,122,162,.15);color:#ff7aa2;border:1px solid rgba(255,122,162,.3);padding:2px 10px;border-radius:10px;font-size:.78rem;font-weight:700;"> → Phase 0</span>'),
}


def generate_html_report(eval_result: dict, cfg: dict, cycle: int = 1) -> Path:
    """
    Phase 3 평가 결과를 HTML 리포트로 생성하고 reports/ 폴더에 저장한다.

    Args:
        eval_result: evaluate_all_channels() 반환값
        cfg:         config.yaml 딕셔너리
        cycle:       Swing Cycle 번호 (1, 2, 3 ...)

    Returns:
        저장된 HTML 파일 경로

    Example:
        >>> from evaluation.metrics import evaluate_all_channels
        >>> from reporting.html_report import generate_html_report
        >>> eval_result = evaluate_all_channels(results, cfg)
        >>> path = generate_html_report(eval_result, cfg, cycle=1)
    """
    reports_dir = Path(cfg["storage"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    prefix   = cfg["reporting"]["html_report_prefix"]
    filename = f"{prefix}_{cycle}.html"
    out_path = reports_dir / filename

    html = _build_html(eval_result, cfg, cycle)
    out_path.write_text(html, encoding="utf-8")

    print(f"  📄  HTML 리포트 저장: {out_path}")
    return out_path


# ──────────────────────────────────────────────
# HTML 빌더
# ──────────────────────────────────────────────
def _build_html(eval_result: dict, cfg: dict, cycle: int) -> str:
    tgt          = cfg["evaluation"]["targets"]
    overall      = eval_result["overall_accuracy"]
    targets_met  = eval_result["targets_met"]
    per_channel  = eval_result["per_channel"]
    swing        = eval_result["swing_decision"]
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_levels   = cfg["data"]["num_levels"]

    status_text  = " 목표 달성" if targets_met else " 목표 미달"
    status_color = "#50e3c2" if targets_met else "#ff7aa2"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grayspot — Swing Cycle {cycle} 평가 리포트</title>
  <style>
    :root {{
      --bg:#0b1220; --fg:#e6eef8; --h1:#66d9ff; --h2:#50e3c2; --h3:#c792ea; --h4:#ff7aa2;
    }}
    * {{ box-sizing:border-box; }}
    body {{ font-family:'Segoe UI',sans-serif; background:var(--bg); color:var(--fg);
            line-height:1.7; margin:0 auto; padding:2rem; max-width:1100px; }}
    h1 {{ color:var(--h1); border-bottom:3px solid rgba(15,109,154,.2);
          padding-bottom:.3rem; font-size:2rem; margin-top:0; }}
    h2 {{ color:var(--h2); border-left:4px solid rgba(80,227,194,.3);
          padding-left:.5rem; font-size:1.4rem; margin-top:2rem; }}
    h3 {{ color:var(--h3); font-size:1.1rem; margin-top:1.4rem; }}
    hr {{ border:none; border-top:1px solid rgba(255,255,255,.1); margin:1.8rem 0; }}
    table {{ border-collapse:collapse; width:100%; margin:.8rem 0; }}
    th,td {{ border:1px solid rgba(255,255,255,.12); padding:.55rem .8rem;
             text-align:left; font-size:.85rem; vertical-align:middle; }}
    th {{ background:rgba(255,255,255,.05); font-weight:700; font-size:.8rem; }}
    tr:hover td {{ background:rgba(255,255,255,.02); }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px; margin:.5rem 0 1.5rem; }}
    .badge {{ font-size:.75rem; padding:3px 11px; border-radius:14px;
              font-weight:600; letter-spacing:.3px; }}
    .b-blue   {{ background:rgba(102,217,255,.1); color:#66d9ff; border:1px solid rgba(102,217,255,.25); }}
    .b-teal   {{ background:rgba(80,227,194,.1);  color:#50e3c2; border:1px solid rgba(80,227,194,.25); }}
    .b-purple {{ background:rgba(199,146,234,.1); color:#c792ea; border:1px solid rgba(199,146,234,.25); }}

    /* 채널 카드 */
    .ch-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:1rem 0; }}
    .ch-card {{ background:rgba(255,255,255,.025); border:1px solid rgba(255,255,255,.08);
                border-radius:12px; padding:16px 18px; position:relative; overflow:hidden; }}
    .ch-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
    .ch-Y::before {{ background:linear-gradient(90deg,#ffeb3b,#f9a825); }}
    .ch-M::before {{ background:linear-gradient(90deg,#f48fb1,#c2185b); }}
    .ch-C::before {{ background:linear-gradient(90deg,#81d4fa,#0277bd); }}
    .ch-K::before {{ background:linear-gradient(90deg,#bdbdbd,#424242); }}
    .ch-title {{ font-size:1rem; font-weight:700; margin-bottom:10px; }}
    .metric-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
    .metric-box {{ background:rgba(11,18,32,.7); border:1px solid rgba(255,255,255,.08);
                   border-radius:8px; padding:8px 14px; text-align:center; flex:1; min-width:70px; }}
    .metric-val {{ font-size:1.2rem; font-weight:700; }}
    .metric-lbl {{ font-size:.68rem; color:rgba(230,238,248,.45); margin-top:1px; }}
    .met   {{ color:#50e3c2; }}
    .unmet {{ color:#ff7aa2; }}

    /* Confusion Matrix */
    .cm-wrap {{ overflow-x:auto; margin-top:8px; }}
    .cm-table {{ width:auto !important; }}
    .cm-table th,
    .cm-table td {{ padding:.35rem .5rem; text-align:center; font-size:.78rem;
                    min-width:38px; border:1px solid rgba(255,255,255,.1); }}
    .cm-table th {{ background:rgba(255,255,255,.06); }}

    /* F1 bar */
    .f1-bar-wrap {{ margin-top:6px; }}
    .f1-row {{ display:flex; align-items:center; gap:8px; margin:3px 0; font-size:.8rem; }}
    .f1-label {{ width:48px; color:rgba(230,238,248,.6); }}
    .f1-track {{ flex:1; background:rgba(255,255,255,.06); border-radius:4px;
                 height:14px; overflow:hidden; }}
    .f1-fill {{ height:100%; border-radius:4px; transition:width .3s; }}
    .f1-val {{ width:38px; text-align:right; color:rgba(230,238,248,.8); }}

    blockquote {{ border-left:4px solid rgba(124,58,237,.6);
                  background:rgba(124,58,237,.08); padding:.7rem 1.1rem;
                  margin:.8rem 0; border-radius:0 6px 6px 0; }}
  </style>
</head>
<body>

<h1>Grayspot 평가 리포트</h1>
<p style="color:rgba(230,238,248,.5);margin-top:.2rem;">
  Swing Cycle {cycle} — Phase 3 Evaluation Report
</p>

<div class="meta">
  <span class="badge b-blue">Cycle {cycle}</span>
  <span class="badge b-teal">{timestamp}</span>
  <span class="badge b-purple" style="color:{status_color};border-color:{status_color}40;">
    {status_text}
  </span>
</div>

<hr>

<!-- 전체 요약 -->
<h2>1. 전체 요약 · Overall Summary</h2>
<table>
  <thead>
    <tr><th>메트릭</th><th>목표값</th><th>실제값</th><th>판정</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Overall Accuracy</td>
      <td>≥ {tgt['overall_accuracy']:.0%}</td>
      <td style="font-weight:700;">{overall:.4f}</td>
      <td>{"<span style='color:#50e3c2'> 달성</span>" if overall >= tgt['overall_accuracy'] else "<span style='color:#ff7aa2'> 미달</span>"}</td>
    </tr>
    {_summary_rows(per_channel, tgt)}
  </tbody>
</table>

<hr>

<!-- 채널별 상세 -->
<h2>2. 채널별 상세 · Per-channel Detail</h2>
<div class="ch-grid">
  {"".join(_channel_card(ch, per_channel[ch], swing[ch], tgt, num_levels) for ch in CHANNELS if ch in per_channel)}
</div>

<hr>

<!-- Swing 판단 -->
<h2>3. Swing 피드백 판단 · Swing Feedback Decision</h2>
<table>
  <thead>
    <tr><th>채널</th><th>Accuracy</th><th>Macro F1</th><th>MAE</th><th>판단</th><th>조치</th></tr>
  </thead>
  <tbody>
    {"".join(_swing_row(ch, per_channel.get(ch, {}), swing.get(ch, "pass")) for ch in CHANNELS)}
  </tbody>
</table>

{_swing_guide(swing)}

</body>
</html>"""


# ──────────────────────────────────────────────
# 헬퍼 함수들
# ──────────────────────────────────────────────
def _summary_rows(per_channel: dict, tgt: dict) -> str:
    rows = ""
    for ch in CHANNELS:
        if ch not in per_channel:
            continue
        m = per_channel[ch]
        acc_ok = m["accuracy"] >= tgt["per_color_accuracy"]
        f1_ok  = m["macro_f1"] >= tgt["per_class_f1"]
        mae_ok = m["mae"]      <= tgt["mae"]
        rows += f"""
    <tr>
      <td>[{ch}] Per-color Accuracy</td>
      <td>≥ {tgt['per_color_accuracy']:.0%}</td>
      <td style="font-weight:700;">{m['accuracy']:.4f}</td>
      <td>{"<span style='color:#50e3c2'></span>" if acc_ok else "<span style='color:#ff7aa2'></span>"}</td>
    </tr>
    <tr>
      <td>[{ch}] Macro F1</td>
      <td>≥ {tgt['per_class_f1']:.2f}</td>
      <td style="font-weight:700;">{m['macro_f1']:.4f}</td>
      <td>{"<span style='color:#50e3c2'></span>" if f1_ok else "<span style='color:#ff7aa2'></span>"}</td>
    </tr>
    <tr>
      <td>[{ch}] MAE</td>
      <td>≤ {tgt['mae']:.2f}</td>
      <td style="font-weight:700;">{m['mae']:.4f}</td>
      <td>{"<span style='color:#50e3c2'></span>" if mae_ok else "<span style='color:#ff7aa2'></span>"}</td>
    </tr>"""
    return rows


def _channel_card(ch: str, m: dict, swing_action: str, tgt: dict, num_levels: int) -> str:
    light, dark = CHANNEL_COLORS[ch]
    acc_cls = "met" if m["accuracy"] >= tgt["per_color_accuracy"] else "unmet"
    f1_cls  = "met" if m["macro_f1"] >= tgt["per_class_f1"]      else "unmet"
    mae_cls = "met" if m["mae"]      <= tgt["mae"]                else "unmet"

    # F1 bar
    f1_bars = ""
    for i, f in enumerate(m.get("per_class_f1", [])):
        pct  = int(f * 100)
        color = "#50e3c2" if f >= tgt["per_class_f1"] else "#ff7aa2"
        f1_bars += f"""
      <div class="f1-row">
        <span class="f1-label">L{i}</span>
        <div class="f1-track">
          <div class="f1-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <span class="f1-val">{f:.2f}</span>
      </div>"""

    # Confusion Matrix
    cm_html = _confusion_matrix_html(m.get("confusion", []), num_levels)

    return f"""
  <div class="ch-card ch-{ch}">
    <div class="ch-title" style="color:{light};">[{ch}] &nbsp; {SWING_BADGE.get(swing_action, swing_action)}</div>
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val {acc_cls}">{m['accuracy']:.3f}</div>
        <div class="metric-lbl">Accuracy</div>
      </div>
      <div class="metric-box">
        <div class="metric-val {f1_cls}">{m['macro_f1']:.3f}</div>
        <div class="metric-lbl">Macro F1</div>
      </div>
      <div class="metric-box">
        <div class="metric-val {mae_cls}">{m['mae']:.3f}</div>
        <div class="metric-lbl">MAE</div>
      </div>
    </div>
    <h3 style="margin:.6rem 0 .3rem;font-size:.85rem;">Per-class F1</h3>
    <div class="f1-bar-wrap">{f1_bars}</div>
    <h3 style="margin:.8rem 0 .3rem;font-size:.85rem;">Confusion Matrix</h3>
    {cm_html}
  </div>"""


def _confusion_matrix_html(cm: list[list[int]], num_levels: int) -> str:
    if not cm:
        return "<p style='color:rgba(230,238,248,.4);font-size:.8rem;'>데이터 없음</p>"

    header = "<tr><th>T\\P</th>" + "".join(f"<th>P{i}</th>" for i in range(num_levels)) + "</tr>"
    rows   = ""
    for i, row in enumerate(cm):
        cells = ""
        total = max(sum(row), 1)
        for j, val in enumerate(row):
            intensity = int(val / total * 180)
            bg = f"rgba(80,227,194,{intensity/255:.2f})" if i == j else \
                 (f"rgba(255,122,162,{intensity/255:.2f})" if val > 0 else "transparent")
            cells += f"<td style='background:{bg};font-weight:{'700' if i==j else '400'};'>{val}</td>"
        rows += f"<tr><th>T{i}</th>{cells}</tr>"

    return f"""<div class="cm-wrap">
      <table class="cm-table">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def _swing_row(ch: str, m: dict, action: str) -> str:
    if not m:
        return f"<tr><td>[{ch}]</td><td colspan='5'>데이터 없음</td></tr>"

    action_desc = {
        "pass":   "목표 달성 — 다음 단계 진행",
        "phase1": "Level 경계 재검토 필요",
        "phase0": "표현 재학습 필요",
    }.get(action, action)

    return f"""<tr>
      <td style="font-weight:700;">[{ch}]</td>
      <td>{m['accuracy']:.4f}</td>
      <td>{m['macro_f1']:.4f}</td>
      <td>{m['mae']:.4f}</td>
      <td>{SWING_BADGE.get(action, action)}</td>
      <td style="font-size:.82rem;color:rgba(230,238,248,.7);">{action_desc}</td>
    </tr>"""


def _swing_guide(swing: dict) -> str:
    needs_phase0 = [ch for ch, a in swing.items() if a == "phase0"]
    needs_phase1 = [ch for ch, a in swing.items() if a == "phase1"]
    all_pass     = all(a == "pass" for a in swing.values())

    if all_pass:
        return """<blockquote>
  <strong> 모든 채널 목표 달성</strong> — 최종 모델 고정 후 배포 진행
</blockquote>"""

    guide = "<blockquote><strong> Swing 복귀 가이드</strong><br>"
    if needs_phase0:
        guide += f"<br>• <strong>Phase 0 재실행 대상</strong>: {', '.join(needs_phase0)} — 표현 학습 방법론 재검토 또는 해당 채널 집중 재학습"
    if needs_phase1:
        guide += f"<br>• <strong>Phase 1 재검토 대상</strong>: {', '.join(needs_phase1)} — Level 경계 재정의 및 혼동 샘플 검토"
    guide += "</blockquote>"
    return guide
