# CI Setup — Continuous Integration / CI 설정

This document describes the GitHub Actions continuous integration pipeline used by the CMYK Grayspot Detection System.
The CI setup enforces code quality, test coverage, build validation, and Docker-based runtime checks to keep the project aligned with the current implementation status.

---

## Table of Contents / 목차

1. [Overview / 개요](#1-overview--개요)
2. [Workflow Triggers / 워크플로우 트리거](#2-workflow-triggers--워크플로우-트리거)
3. [Pipeline Jobs / 파이프라인 작업](#3-pipeline-jobs--파이프라인-작업)
4. [Test Matrix / 테스트 매트릭스](#4-test-matrix--테스트-매트릭스)
5. [Artifacts and Reporting / 아티팩트 및 보고](#5-artifacts-and-reporting--아티팩트-및-보고)
6. [Local Validation / 로컬 검증]
7. [Maintenance / 유지보수]

---

## 1. Overview / 개요

The CI pipeline is defined in `.github/workflows/ci.yml`.
It is designed to verify that repository changes meet project quality requirements before merge.

The current CI pipeline covers:

- linting and formatting
- unit, integration, and smoke tests
- package/build validation
- Docker image build and runtime smoke check
- cross-platform Python version validation

이 CI 파이프라인은 프로젝트 변경 사항이 병합되기 전에 다음을 검증합니다.

- 린팅 및 코드 포맷
- 단위, 통합, 스모크 테스트
- 패키지/빌드 유효성 검증
- Docker 이미지 빌드 및 실행 검증
- 교차 플랫폼 Python 버전 검증

---

## 2. Workflow Triggers / 워크플로우 트리거

The workflow runs on:

- `push` to any branch
- `pull_request` targeting any branch
- manual dispatch via `workflow_dispatch`

이 워크플로우는 다음 이벤트에서 실행됩니다.

- 모든 브랜치에 대한 `push`
- 모든 브랜치 대상 `pull_request`
- 수동 실행 (`workflow_dispatch`)

---

## 3. Pipeline Jobs / 파이프라인 작업

The CI pipeline defines four jobs.
Each job is responsible for a distinct validation stage.

### 3.1 `lint`

- `actions/checkout@v4`
- `actions/setup-python@v5` with Python 3.11
- installs `flake8`, `black`, `isort`, and `mypy`
- runs strict flake8 checks on `src`
- formats code with `black` and `isort`
- commits formatting updates automatically via `stefanzweifel/git-auto-commit-action@v5`

### 3.2 `test`

- depends on `lint`
- uses a matrix over OS and Python versions
- installs dependencies from `requirements.txt`
- runs:
  - unit tests `src/tests/unit`
  - integration tests `src/tests/integration` excluding slow tests
  - smoke tests `src/tests/smoke` with `continue-on-error: true`
- uploads coverage and test artifacts

### 3.3 `build`

- installs build dependencies (`build`, `wheel`)
- validates `src/config/pyproject.toml` parsing
- checks that `requirements.txt` is structurally valid

### 3.4 `docker`

- depends on `test`
- builds the Docker image `grayspot:test`
- runs a basic runtime check inside the image:
  - `python -c "import torch; print('PyTorch version:', torch.__version__)"`

---

## 4. Test Matrix / 테스트 매트릭스

The `test` job runs across the following environment matrix:

| OS | Python Version | Coverage |
|---|---|---|
| `ubuntu-latest` | `3.10` | yes |
| `ubuntu-latest` | `3.11` | yes |
| `windows-latest` | `3.11` | yes |
| `macos-latest` | `3.11` | yes |

The matrix provides a broad compatibility check for supported developer platforms.

이 매트릭스는 지원되는 개발 플랫폼에 대한 호환성 검사를 제공합니다.

---

## 5. Artifacts and Reporting / 아티팩트 및 보고

The CI pipeline currently produces and uploads:

- `coverage.xml` from pytest coverage
- `.pytest_cache/` artifacts for debugging test runs
- a Docker image build validation report

Coverage is uploaded via `codecov/codecov-action@v4` with `fail_ci_if_error: false`.
This means coverage upload failures do not fail the pipeline, but underlying test failures still do.

CI는 다음을 업로드합니다.

- pytest 커버리지 결과 (`coverage.xml`)
- `.pytest_cache/` 디버깅 아티팩트
- Docker 빌드/실행 검증 결과

`codecov` 업로드 실패는 CI 실패로 처리되지 않도록 설정되어 있습니다.

---

## 6. Local Validation / 로컬 검증

Before pushing changes, execute the following locally to mirror CI behavior:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest src/tests/unit --cov=src --cov-report=xml
python -m pytest src/tests/integration -m "not slow"
python -m pytest src/tests/smoke -m smoke
```

Optional local formatting check:

```bash
python -m isort src --profile black
python -m black src
```

---

## 7. Maintenance / 유지보수

When CI requirements change:

- update `.github/workflows/ci.yml`
- update `requirements.txt` or `src/config/pyproject.toml` as needed
- document the change in `doc/CI_Setup.md`
- verify the new workflow by creating a temporary branch or pull request

CI 요구 사항이 변경되면:

- `.github/workflows/ci.yml` 업데이트
- 필요 시 `requirements.txt` 또는 `src/config/pyproject.toml` 업데이트
- 변경 내용을 `doc/CI_Setup.md`에 문서화
- 임시 브랜치 또는 PR로 새 워크플로우 검증

---

## 8. Current Project Status / 현재 프로젝트 상태

The project currently has an active GitHub Actions CI configuration with:

- enforced formatting and linting for source code
- multi-platform Python test coverage
- lightweight Docker runtime verification
- automatic style commit support

이 프로젝트는 다음을 포함하는 GitHub Actions CI 구성을 실행 중입니다.

- 소스 코드 포맷 및 린팅 강제
- 다중 플랫폼 Python 테스트 커버리지
- 경량 Docker 런타임 검증
- 자동 스타일 커밋 지원
