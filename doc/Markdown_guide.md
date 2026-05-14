# Markdown Guide — CMYK Printer Project

This document explains the markdown conventions and structure used throughout the CMYK Printer Grayspot Detection Project.

이 문서는 CMYK 프린터 Grayspot 탐지 프로젝트 전체에서 사용되는 마크다운 규칙 및 구조를 설명합니다.

---

## 1. Heading Hierarchy / 제목 계층 구조

The project uses standard Markdown heading levels to organize documentation clearly.

```markdown
# Main Title / 메인 제목                    [PRD, README 최상위]
## Section / 섹션                           [장(Chapter)]
### Subsection / 하위섹션                   [소주제(Topic)]
#### Detailed Topics / 상세주제             [세부항목(Subtopic)]
##### Minor points / 부분항목              [부분 항목]
```

### Examples / 예시

- **README.md**: Uses `#` for title, `##` for major sections (Installation, Configuration)
- **PRD Documents**: Uses hierarchical structure to separate concerns
- **Code Documentation**: Uses `##` for module section, `###` for class/function groups

---

## 2. Content Structure / 콘텐츠 구조

### 2.1 Documentation Files / 문서 파일

Each documentation file should follow this structure:

```markdown
# Document Title / 문서 제목

Brief description in English and Korean

---

## Table of Contents / 목차

1. [Section 1](#section-1)
2. [Section 2](#section-2)

---

## Section 1

Content here...

### Subsection 1.1

More detailed content...

---

## Section 2

...
```

### 2.2 Code Documentation / 코드 문서

Python modules include docstrings with the following structure:

```python
"""
module/file.py

Brief description in English and Korean (한글).

Features / 기능:
    - Feature 1 description
    - Feature 2 description

Usage / 사용법:
    Example code blocks with comments

Python 3.11.5 | PyTorch 2.x compatible
"""
```

---

## 3. Formatting Conventions / 포맷 규칙

### 3.1 Code Blocks / 코드 블록

Use triple backticks with language specification:

````markdown
```python
# Python code
def function():
    pass
```

```bash
# Bash/Shell commands
python src/scripts/train.py
```

```yaml
# YAML configuration
system:
  device: "auto"
```
````

### 3.2 Lists / 목록

Use consistent bullet formatting:

```markdown
- Item 1
- Item 2
  - Nested item 2.1
  - Nested item 2.2
- Item 3

1. Ordered item 1
2. Ordered item 2
   1. Nested ordered item 2.1
   2. Nested ordered item 2.2
```

### 3.3 Emphasis / 강조

```markdown
**Bold text** for important terms
*Italic text* for file names or emphasis
`code` for inline code references
```

### 3.4 Links / 링크

```markdown
[Display Text](url)
[GitHub](https://github.com)
[Section Reference](#section-name)

For local files:
[README](README.md)
[Config](src/config/config.yaml)
```

### 3.5 Tables / 표

```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

| Component | Description | Status |
|-----------|-------------|--------|
| Training  | Training pipeline | ✅ Complete |
| Inference | Inference module | ✅ Complete |
| Reports   | HTML report generation | ✅ Complete |
```

---

## 4. Project-Specific Conventions / 프로젝트별 규칙

### 4.1 Bilingual Formatting / 이중 언어 포맷

All documents should provide both English and Korean explanations:

```markdown
## Section Title / 섹션 제목

English explanation...

한국어 설명...

### Subsection / 하위섹션

English details

한국어 세부사항
```

### 4.2 Feature Lists / 기능 목록

When listing features, always provide both languages:

```markdown
## Features / 기능

### English Version
- Feature 1: Description
- Feature 2: Description

### 한국어 버전
- 기능 1: 설명
- 기능 2: 설명

Or combined format:
- Batch inference support / 배치 추론 지원
- Auto device detection (CUDA/MPS/CPU) / 자동 장치 감지
- Cached model loading / 캐시된 모델 로딩
```

### 4.3 Status Indicators / 상태 표시

Use emoji for quick status reference:

```markdown
✅ Complete / 완료
🚧 In Progress / 진행 중
❌ Not Started / 미시작
⚠️ Warning / 경고
💡 Tip / 팁
📝 Note / 참고
```

### 4.4 Version Information / 버전 정보

Include version info at the end of major documents:

```markdown
---

```

---

## 5. README Best Practices / README 작성 규칙

### Structure / 구조

```markdown
# Project Title

Brief description (1-2 sentences)

---

## Table of Contents
## System Requirements
## Installation
## Quick Start
## Project Structure
## Configuration
## Usage / Training / Inference
## Docker Usage
## Troubleshooting
## Version History
## Contributing
## License
## Support
```

### Installation Section / 설치 섹션

Always include:
- Minimum requirements
- Recommended requirements
- Platform-specific instructions
- Virtual environment setup
- Docker alternative

### Examples / 예시

Provide actual working examples:

```markdown
### Example: Training

\`\`\`bash
python src/scripts/train.py --channel Y
\`\`\`

### Example: Inference

\`\`\`python
from src.inference.predictor import GrayspotPredictor
predictor = GrayspotPredictor()
\`\`\`
```

---

## 6. Documentation Structure for Modules / 모듈 문서화 구조

### For Python Modules / Python 모듈의 경우

```python
"""
module_name.py

One-line description / 한 줄 설명

Detailed description in both English and Korean.
영어와 한글 모두의 상세 설명.

Features / 기능:
    - Feature 1 / 기능 1
    - Feature 2 / 기능 2

Classes / 클래스:
    ClassName: Brief description

Functions / 함수:
    function_name: Brief description

Usage / 사용법:
    >>> from module import ClassName
    >>> obj = ClassName()

Python 3.10+ | Compatible with PyTorch 2.x
"""
```

---

## 7. Common Patterns / 일반적인 패턴

### Installation Instructions / 설치 지시사항

```markdown
### Installation / 설치

#### Local Setup / 로컬 설정

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

#### Docker Setup / Docker 설정

\`\`\`bash
docker build -t project:latest .
docker run --rm -it project:latest
\`\`\`
```

### Platform-Specific Instructions / 플랫폼별 지시사항

```markdown
### PyTorch Installation / PyTorch 설치

#### macOS — Apple Silicon (MPS)
\`\`\`bash
pip install torch torchvision
\`\`\`

#### Windows / Linux — GPU (CUDA 11.8)
\`\`\`bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
\`\`\`

#### Windows / Linux — CPU Only
\`\`\`bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
\`\`\`
```

---

## 8. File Organization / 파일 조직

```
docs/
├── README.md                    # Main project documentation
├── Markdown_guide.md            # This file
├── Client_Q.md                  # Client questions & answers
├── S1_review.md                 # S1 review notes
├── CONFIGURATION_OPTIMIZATION_SUMMARY.md
└── OPTIMIZATION_COMPLETE.md

src/
├── module/
│   ├── __init__.py             # Module docstring
│   ├── submodule.py            # File with module-level docstring
│   └── README.md               # Optional module-level README
```

---

## 9. Validation Checklist / 검증 체크리스트

When writing documentation, verify:

- ✅ Consistent heading hierarchy
- ✅ Bilingual content (English + Korean) where applicable
- ✅ Code blocks with proper syntax highlighting
- ✅ Working examples included
- ✅ Proper formatting and spacing
- ✅ Links to related sections work
- ✅ No broken references
- ✅ Version information up-to-date
- ✅ Status indicators clear
- ✅ Installation instructions tested

---

## Quick Reference / 빠른 참조

| Element | Markdown | Purpose |
|---------|----------|---------|
| Heading | `# Title` | Document sections |
| Bold | `**text**` | Emphasis |
| Code | `` `code` `` | Inline code |
| Code Block | ` ``` ` | Multi-line code |
| Link | `[text](url)` | References |
| List | `- item` | Bullet points |
| List | `1. item` | Numbered list |
| Table | `\| col \|` | Structured data |
| Horizontal Rule | `---` | Section breaks |
| Status | `✅ ✓` | Quick indicators |

---


Example table format / 표 형식 예시:

| Metric / 메트릭 | Target / 목표값 | Notes / 비고 |
|--------|--------|-------|
| Overall Accuracy / 전체 정확도 | ≥ 90% | All colors combined / 전체 색상 통합 |
| Per-class F1 Score / 클래스별 F1 점수 | ≥ 0.80 | Per Level (0–5) / 레벨별 (0~5) |

This is useful when the document needs to compare multiple items or define requirements in a compact form.
여러 항목을 비교하거나 요구사항을 간결하게 정의해야 할 때 유용합니다.

---

### 5. Bullet Lists and Numbered Lists / 불릿 목록 및 번호 목록

The document uses both unordered and ordered lists.
문서는 순서 없는 목록과 순서 있는 목록을 모두 사용합니다.

#### Bullet Lists / 불릿 목록
Used for / 다음 용도로 사용됩니다:
- design principles / 설계 원칙
- feature descriptions / 기능 설명
- module responsibilities / 모듈 책임
- policy summaries / 정책 요약

Example / 예시:

- Training and Inference share the same preprocessing standard / 학습과 추론은 동일한 전처리 표준을 공유합니다
- ROI segmentation defaults to automatic boundary detection / ROI 분할은 자동 경계 검출을 기본으로 합니다
- Fixed-coordinate fallback may be used in early implementation / 초기 구현 단계에서는 고정 좌표 방식을 병행할 수 있습니다

#### Numbered Lists / 번호 목록
Used for / 다음 용도로 사용됩니다:
- procedures / 절차
- workflows / 워크플로우
- implementation steps / 구현 단계
- instructions / 지침

Example / 예시:

1. Print CMYK test patterns / CMYK 테스트 패턴 출력
2. Scan the printout / 출력물 스캔
3. Save the file using the naming convention / 파일명 규칙에 따라 저장
4. Apply preprocessing / 전처리 적용
5. Train the model / 모델 학습

This makes the document practical for both explanation and execution.
이는 문서를 설명과 실행 모두에 실용적으로 만들어 줍니다.

---

### 6. Blockquotes for Notes and Warnings / 노트 및 경고를 위한 인용문

The PRD uses blockquotes (`>`) to highlight important notes, cautions, or design decisions.
PRD는 인용문(`>`)을 사용하여 중요한 노트, 주의사항 또는 설계 결정 사항을 강조합니다.

Example / 예시:

> **Note / 참고**: The industry-standard term **CMYK** is used for the color space designation. / 색상 공간 명칭으로 업계 표준 용어 **CMYK**를 사용합니다.
> **Fallback on underperformance / 성능 미달 시 대응**: Apply the fallback strategy described in Section 14. / Section 14에 설명된 Fallback 전략을 적용합니다.

This is useful when certain information should stand out from the main paragraph content.
특정 정보가 본문 내용과 구별되어 눈에 띄어야 할 때 유용합니다.

---

### 7. Horizontal Rules / 수평 구분선

The document uses horizontal separators (`---`) to visually divide large content blocks.
문서는 수평 구분선(`---`)을 사용하여 큰 내용 블록을 시각적으로 구분합니다.

Example / 예시:

---

These separators improve readability, especially in long technical documents.
이 구분선은 특히 긴 기술 문서에서 가독성을 향상시킵니다.

---

### 8. Fenced Code Blocks / 코드 블록

The PRD includes fenced code blocks for structured content that should remain unformatted as plain text.
PRD는 일반 텍스트로 서식 없이 유지되어야 하는 구조화된 내용을 위해 코드 블록을 포함합니다.

These are used for / 다음 용도로 사용됩니다:

- directory structures / 디렉토리 구조
- JSON examples / JSON 예시
- YAML examples / YAML 예시
- pseudo workflows / 의사 워크플로우
- layout sketches / 레이아웃 스케치
- technical configuration examples / 기술 설정 예시

Example / 예시:

```json
{
  "version": "2.1.0",
  "architecture": "EfficientNet-B0",
  "training_accuracy": 0.92
}
```

Example / 예시:
```yaml
overall_policy:
  method: max_level
  warning_threshold: 3
  bad_threshold: 4
```

Code blocks are important because they preserve formatting and make technical examples easier to understand.
코드 블록은 서식을 보존하고 기술적인 예시를 더 쉽게 이해할 수 있게 만들기 때문에 중요합니다.