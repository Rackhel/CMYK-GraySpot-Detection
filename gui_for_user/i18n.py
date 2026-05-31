"""gui_for_user 다국어 지원 / i18n module.
사용법: from gui_for_user.i18n import t, set_lang, get_lang
"""

from __future__ import annotations

_lang: str = "ko"

_STRINGS: dict[str, dict[str, str]] = {
    # ── App / Window ──────────────────────────────────────────────────────────
    "app_title": {"ko": "CMYK Level Inspector", "en": "CMYK Level Inspector"},
    "header_hint": {
        "ko": "이미지를 드래그하거나 버튼으로 선택하세요",
        "en": "Drag an image here or use the button below",
    },
    # ── Sidebar ───────────────────────────────────────────────────────────────
    "sidebar_title": {"ko": "⚙  모델 설정", "en": "⚙  Model Settings"},
    "grp_checkpoints": {"ko": "가중치 파일", "en": "Checkpoints"},
    "grp_settings": {"ko": "추론 설정", "en": "Inference Settings"},
    "lbl_device": {"ko": "Device", "en": "Device"},
    "lbl_ch_mode": {"ko": "채널 모드", "en": "Channel Mode"},
    "mode_ensemble": {"ko": "전체 앙상블  Y+M+C+K", "en": "Ensemble  Y+M+C+K"},
    "mode_single_ch": {"ko": "{ch} 채널만", "en": "{ch} channel only"},
    "btn_apply": {"ko": "✅  적용", "en": "✅  Apply"},
    "btn_detect_all": {"ko": "🔍  전체 자동 탐지", "en": "🔍  Auto-detect All"},
    "ckpt_placeholder": {
        "ko": "(비어 있으면 자동 탐지)",
        "en": "(auto-detect if empty)",
    },
    "tooltip_browse": {"ko": "파일 탐색", "en": "Browse file"},
    "tooltip_auto": {"ko": "자동 탐지", "en": "Auto-detect"},
    "tooltip_detect_all": {
        "ko": "models_dir 에서 4채널 체크포인트를 자동으로 찾습니다",
        "en": "Scan models_dir for all 4-channel checkpoints",
    },
    # ── Toolbar ───────────────────────────────────────────────────────────────
    "btn_select_img": {"ko": "🖼  이미지 선택", "en": "🖼  Select Image"},
    "btn_run": {"ko": "▶  추론 실행", "en": "▶  Run Inference"},
    "btn_stop": {"ko": "■  중지", "en": "■  Stop"},
    "toolbar_placeholder": {
        "ko": "파일을 선택하거나 드래그하세요",
        "en": "Select or drag an image here",
    },
    # ── Result panel ──────────────────────────────────────────────────────────
    "lbl_preview": {"ko": "미리보기", "en": "Preview"},
    "preview_hint": {
        "ko": "이미지를 선택하거나\n드래그 & 드롭 하세요",
        "en": "Select an image\nor drag & drop here",
    },
    "lbl_ch_result": {"ko": "채널별 예측 레벨", "en": "Per-channel Level"},
    "lbl_ensemble": {"ko": "앙상블 결과", "en": "Ensemble Result"},
    "lbl_top3": {"ko": "Top-3 예측", "en": "Top-3 Prediction"},
    "lbl_status": {"ko": "상태", "en": "Status"},
    "status_ready": {"ko": "대기 중", "en": "Ready"},
    "status_main": {"ko": "준비", "en": "Ready"},
    # ── Dialog titles ─────────────────────────────────────────────────────────
    "dlg_select_img": {"ko": "이미지 선택", "en": "Select Image"},
    "dlg_select_ckpt": {"ko": "가중치 파일 선택", "en": "Select Checkpoint"},
    # ── Log messages ──────────────────────────────────────────────────────────
    "log_done": {"ko": "✅ 완료", "en": "✅ Done"},
    "log_error": {"ko": "❌ 오류", "en": "❌ Error"},
    "settings_applied": {"ko": "설정 적용됨", "en": "Settings applied"},
    "mode_label_ens": {"ko": "앙상블", "en": "Ensemble"},
    "mode_label_ch": {"ko": "{ch} 채널", "en": "{ch} channel"},
    # ── Language toggle ───────────────────────────────────────────────────────
    "btn_lang": {"ko": "🌐 EN", "en": "🌐 한국어"},
    "tooltip_toggle": {"ko": "사이드바 열기 / 닫기", "en": "Toggle sidebar"},
}


def set_lang(lang: str) -> None:
    global _lang
    if lang in ("ko", "en"):
        _lang = lang


def get_lang() -> str:
    return _lang


def t(key: str, **kwargs) -> str:
    """현재 언어로 키를 번역한다. kwargs는 .format()에 전달된다."""
    row = _STRINGS.get(key)
    if row is None:
        return key
    s = row.get(_lang, row.get("ko", key))
    return s.format(**kwargs) if kwargs else s
