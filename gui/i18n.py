"""GUI 다국어 지원 / GUI internationalization.

사용법 / Usage:
    from gui.i18n import t, set_lang, get_lang
    label.setText(t("save"))
"""

from __future__ import annotations

_LANG: str = "ko"  # "ko" | "en"

# ── 번역 테이블 / Translation table ──────────────────────────────────────────
_T: dict[str, dict[str, str]] = {
    # ── Tab labels ────────────────────────────────────────────────────────
    "tab_data": {"ko": "데이터", "en": "Data"},
    "tab_training": {"ko": "학습", "en": "Training"},
    "tab_evaluation": {"ko": "평가", "en": "Evaluation"},
    "tab_settings": {"ko": "설정", "en": "Settings"},
    "tab_optuna": {"ko": "Optuna HPO", "en": "Optuna HPO"},
    "tab_embedding": {"ko": "임베딩", "en": "Embedding"},
    "tab_inference": {"ko": "추론", "en": "Inference"},
    # ── Group titles ──────────────────────────────────────────────────────
    "grp_storage": {"ko": "저장 경로", "en": "Storage / Paths"},
    "grp_phase2": {
        "ko": "Phase 2 — 지도 학습",
        "en": "Phase 2 — Supervised Classification",
    },
    "grp_phase0": {
        "ko": "Phase 0 — SimCLR 대조 학습",
        "en": "Phase 0 — SimCLR Contrastive Learning",
    },
    "grp_train_cmn": {"ko": "학습 (공통)", "en": "Training (Common)"},
    "grp_appearance": {"ko": "화면 설정", "en": "Appearance"},
    "grp_model_cfg": {"ko": "모델 설정", "en": "Model Configuration"},
    "grp_run_train": {"ko": "학습 실행", "en": "Run Training"},
    "grp_eval": {"ko": "데이터셋 평가", "en": "Dataset Evaluation"},
    "grp_infer": {"ko": "단일 이미지 추론", "en": "Single Image Inference"},
    "grp_ss_editor": {
        "ko": "── 탐색 공간 편집기 ──────────────────────",
        "en": "── Search Space Editor ──────────────────────",
    },
    "grp_single_infer": {"ko": "단일 이미지 추론", "en": "Single Image Inference"},
    "grp_batch_infer": {"ko": "배치 폴더 추론", "en": "Batch Folder Inference"},
    "grp_ckpt": {"ko": "체크포인트", "en": "Checkpoint"},
    "grp_result": {"ko": "추론 결과", "en": "Inference Result"},
    # ── Buttons ───────────────────────────────────────────────────────────
    "btn_save_settings": {"ko": "설정 저장", "en": "Save Settings"},
    "btn_reset": {"ko": "현재 설정으로 초기화", "en": "Reset to Current Config"},
    "btn_start_training": {"ko": "▶  학습 시작", "en": "▶  Start Training"},
    "btn_stop": {"ko": "■  중지", "en": "■  Stop"},
    "btn_save_model": {"ko": "모델 설정 저장", "en": "Save Model Config"},
    "btn_run_eval": {"ko": "▶  평가 실행", "en": "▶  Run Evaluation"},
    "btn_browse_img": {"ko": "📂  이미지 선택…", "en": "📂  Browse Image…"},
    "btn_run_infer": {"ko": "▶  추론 실행", "en": "▶  Run Inference"},
    "btn_scan": {"ko": "데이터셋 스캔", "en": "Scan Dataset"},
    "btn_browse": {"ko": "이미지 선택…", "en": "Browse Image…"},
    "btn_start_hpo": {"ko": "▶  HPO 시작", "en": "▶  Start HPO"},
    "btn_save_ss": {
        "ko": "탐색 공간 저장 → config.json",
        "en": "Save Search Space → config.json",
    },
    "btn_extract_emb": {"ko": "▶  임베딩 추출", "en": "▶  Extract Embeddings"},
    "btn_save_correction": {
        "ko": "💾  라벨 교정 저장",
        "en": "💾  Save Label Correction",
    },
    "btn_browse_ckpt": {"ko": "📂  체크포인트 선택…", "en": "📂  Browse Checkpoint…"},
    "btn_auto_detect": {"ko": "🔍  자동 탐지", "en": "🔍  Auto-detect"},
    "btn_browse_folder": {"ko": "📂  폴더 선택…", "en": "📂  Browse Folder…"},
    "btn_run_batch": {"ko": "▶  배치 추론", "en": "▶  Run Batch Inference"},
    "btn_export_csv": {"ko": "💾  CSV 내보내기", "en": "💾  Export CSV"},
    # ── Labels ────────────────────────────────────────────────────────────
    "lbl_theme": {"ko": "테마", "en": "Theme"},
    "lbl_lang": {"ko": "언어", "en": "Language"},
    "lbl_channel": {"ko": "채널", "en": "Channel"},
    "lbl_phase": {"ko": "Phase", "en": "Phase"},
    "lbl_trials": {"ko": "Trials", "en": "Trials"},
    "lbl_new_level": {"ko": "교정 레벨", "en": "New Level"},
    "lbl_ds_overview": {
        "ko": "데이터셋 현황 — 채널 × 레벨 샘플 수",
        "en": "Dataset Overview — Channel × Level Sample Count",
    },
    "lbl_img_preview": {"ko": "이미지 미리보기", "en": "Image Preview"},
    "lbl_selected": {"ko": "선택됨: (없음)", "en": "Selected: (none)"},
    "lbl_pred": {"ko": "예측 레벨: —", "en": "Predicted Level: —"},
    "lbl_conf": {"ko": "신뢰도: —", "en": "Confidence: —"},
    "lbl_top3": {"ko": "Top-3: —", "en": "Top-3: —"},
    "lbl_no_ckpt": {"ko": "체크포인트 미지정", "en": "No checkpoint selected"},
    "lbl_no_folder": {"ko": "폴더 미지정", "en": "No folder selected"},
    "lbl_conf_val": {"ko": "신뢰도: {v}", "en": "Confidence: {v}"},
    "col_filename": {"ko": "파일명", "en": "Filename"},
    "col_pred_level": {"ko": "예측 레벨", "en": "Pred. Level"},
    "col_confidence": {"ko": "신뢰도", "en": "Confidence"},
    # ── Theme names ───────────────────────────────────────────────────────
    "theme_dark": {"ko": "🌙  다크", "en": "🌙  Dark"},
    "theme_light": {"ko": "☀️  라이트", "en": "☀️  Light"},
}


def t(key: str) -> str:
    """현재 언어로 번역된 문자열을 반환한다."""
    return _T.get(key, {}).get(_LANG, key)


def set_lang(lang: str) -> None:
    """언어를 설정한다 / Set active language. lang: 'ko' | 'en'"""
    global _LANG
    if lang in ("ko", "en"):
        _LANG = lang


def get_lang() -> str:
    return _LANG
