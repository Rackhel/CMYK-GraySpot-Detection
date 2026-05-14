"""
reporting/__init__.py

보고서 생성 패키지 / Reporting package.

HTML 기반 평가 보고서 생성 모듈을 내보낸다.
Exports HTML-based evaluation report generation modules.

주요 모듈 / Key module:
    html_report : 평가 결과 → HTML 보고서 렌더링
                  Renders evaluation results into an HTML report

사용법 / Usage:
    from reporting.html_report import generate_html_report
"""

# html_report 는 동적 임포트 허용 (무거운 의존성: plotly)
# html_report allows lazy import (heavy dependency: plotly)
from . import html_report  # noqa: F401

__all__ = [
    "html_report",
]
