## PRD Document Markdown Structure Explanation
## PRD 문서 마크다운 구조 설명

The PRD document is organized using a standard Markdown hierarchy combined with several practical documentation elements. The overall structure can be understood as follows:
PRD 문서는 표준 마크다운 계층 구조와 여러 실용적인 문서 요소를 결합하여 구성되어 있습니다. 전체 구조는 다음과 같이 이해할 수 있습니다.

### 1. Title Hierarchy / 제목 계층 구조

The document uses Markdown heading levels to define its hierarchy clearly.
문서는 마크다운 제목 레벨을 사용하여 계층 구조를 명확하게 정의합니다.

- `#` : Main document title / 메인 문서 제목
- `##` : Major sections / chapters / 주요 섹션 / 챕터
- `###` : Subsections / 하위 섹션
- `####` : Detailed subtopics / 세부 하위 주제

Example / 예시:

# Grayspot Detection Pipeline Design
## Full Pipeline Design Document
## 1. System Overview
### 1.1 Objective
#### Learning Flow

This structure makes the document easy to scan and helps readers understand the relationship between high-level sections and detailed implementation notes.
이 구조는 문서를 쉽게 훑어볼 수 있게 하며, 상위 섹션과 세부 구현 내용 간의 관계를 독자가 이해하는 데 도움을 줍니다.

---

### 2. Major Section Structure / 주요 섹션 구조

The document is divided into large sections using `##` headings.
문서는 `##` 제목을 사용하여 큰 섹션으로 나뉩니다.

Each major section represents one important area of the PRD, such as:
각 주요 섹션은 PRD의 중요한 영역 하나를 나타내며, 예를 들면 다음과 같습니다.

- System Overview / 시스템 개요
- Overall Pipeline Architecture / 전체 파이프라인 아키텍처
- Training Pipeline / 학습 파이프라인
- Module Structure / 모듈 구조
- Core Module Detailed Design / 핵심 모듈 상세 설계
- Data Collection and Labeling Guide / 데이터 수집 및 라벨링 가이드
- Implementation Steps / 구현 단계
- Technology Stack / 기술 스택
- GUI Design / GUI 설계
- Experiment Tracking Strategy / 실험 추적 전략
- Data Drift Monitoring Strategy / 데이터 드리프트 모니터링 전략
- Confidence-Driven Operational Policy / 신뢰도 기반 운영 정책
- Appendices / 부록

This means the PRD is not written as one long narrative, but as a structured technical design document with clearly separated topics.
즉, PRD는 하나의 긴 서술 형식이 아니라 명확하게 구분된 주제를 가진 구조화된 기술 설계 문서로 작성됩니다.

---

### 3. Subsection Structure / 하위 섹션 구조

Within each major section, the document uses `###` and sometimes `####` headings to break content into smaller logical units.
각 주요 섹션 내에서 문서는 `###` 및 경우에 따라 `####` 제목을 사용하여 내용을 더 작은 논리적 단위로 나눕니다.

For example / 예를 들면:

- `### 1.1 Objective / 목표`
- `### 1.2 Input / Output Definition / 입출력 정의`
- `### 1.3 Grayspot Level Definition / Grayspot 레벨 정의`
- `### 1.4 Performance Targets / 성능 목표`

When additional detail is needed, `####` is used for deeper explanation.
추가적인 세부 사항이 필요한 경우 `####`을 사용하여 더 깊은 설명을 제공합니다.

Example / 예시:

- `#### Architecture Configuration / 아키텍처 구성`
- `#### Candidate Methodologies / 후보 방법론`
- `#### Positive Pair Definition / Positive Pair 정의`

So the basic structure is / 기본 구조는 다음과 같습니다:

- `##` = chapter / 챕터
- `###` = topic inside the chapter / 챕터 내 주제
- `####` = detailed explanation inside the topic / 주제 내 세부 설명

---

### 4. Tables / 표

A large portion of the PRD uses tables to present structured information clearly.
PRD의 많은 부분은 구조화된 정보를 명확하게 표현하기 위해 표를 사용합니다.

Tables are used for / 표는 다음 용도로 사용됩니다:

- input/output definitions / 입출력 정의
- performance targets / 성능 목표
- hyperparameters / 하이퍼파라미터
- module responsibilities / 모듈 책임
- evaluation metrics / 평가 메트릭
- policies / 정책
- decision criteria / 의사결정 기준

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