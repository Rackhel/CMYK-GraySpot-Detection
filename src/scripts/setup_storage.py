"""
Grayspot — 로컬 스토리지 초기화 / Local Storage Initialization
setup_storage.py

실행 / Run: python setup_storage.py

download_dataset.py 실행 후 이 스크립트를 실행하면
나머지 폴더 구조가 자동으로 생성된다.

Run this script after download_dataset.py to automatically
create the remaining folder structure.
"""

from pathlib import Path
import yaml


def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """config.yaml을 로드한다. / Loads config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_storage(cfg: dict) -> None:
    """
    프로젝트 전체 폴더 구조를 생성한다.
    이미 존재하는 폴더는 그대로 두고 없는 것만 생성한다.

    Creates the full project folder structure.
    Existing folders are left untouched; only missing ones are created.
    """
    st         = cfg["storage"]
    channels   = cfg["data"]["channels"]    # ["Y", "M", "C", "K"]
    num_levels = cfg["data"]["num_levels"]  # 6 (Level 0~5)

    folders: list[Path] = []

    # raw/ — download_dataset.py가 data/images/ 에 저장하므로 폴더만 확인
    # raw/ — download_dataset.py saves to data/images/, just ensure folder exists
    folders.append(Path(st["raw_dir"]))

    # labeled/ — CMYK 채널 × Level 0~5 / CMYK channels × Level 0~5
    for ch in channels:
        for lv in range(num_levels):
            folders.append(Path(st["labeled_dir"]) / ch / f"level_{lv}")

    # training/ — Phase 0 (Contrastive) / Phase 2 (Supervised)
    folders.append(Path(st["training_dir"]) / "phase0")
    folders.append(Path(st["training_dir"]) / "phase2")

    # analyzed/ — 추론 결과 JSON / Inference result JSONs
    folders.append(Path(st["analyzed_dir"]))

    # reports/ — 평가 리포트 및 로그 / Evaluation reports and logs
    folders.append(Path(st["reports_dir"]))
    folders.append(Path(st["reports_dir"]) / "logs")

    # models/ — 학습된 모델 가중치 / Trained model weights
    folders.append(Path(cfg["inference"]["model_dir"]))

    # 폴더 생성 / Create folders
    created, skipped = 0, 0
    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATE] {folder}")
            created += 1
        else:
            print(f"  [SKIP]   {folder}")
            skipped += 1

    # 빈 폴더에 .gitkeep 배치 (Git 추적용) / Place .gitkeep in empty folders (for Git tracking)
    _place_gitkeep(Path(st["base_dir"]))

    print(f"\n  Done — {created} created, {skipped} skipped")
    print(f"📂  Root: {Path(st['base_dir']).resolve()}\n")
    print_tree(Path(st["base_dir"]))


def _place_gitkeep(base_dir: Path) -> None:
    """
    빈 폴더마다 .gitkeep 파일을 생성한다.
    Git은 빈 폴더를 추적하지 않으므로 .gitkeep으로 폴더 존재를 보장한다.

    Creates .gitkeep files in empty folders.
    Git does not track empty folders, so .gitkeep ensures folder existence.
    """
    for folder in base_dir.rglob("*"):
        if folder.is_dir():
            if not any(f for f in folder.iterdir() if f.name != ".gitkeep"):
                (folder / ".gitkeep").touch()


def print_tree(base_dir: Path, indent: str = "", max_depth: int = 4, _depth: int = 0) -> None:
    """
    폴더 트리를 터미널에 출력한다. / Prints the folder tree to the terminal.
    .gitkeep 파일은 표시하지 않는다. / .gitkeep files are not displayed.
    """
    if _depth > max_depth:
        return
    entries = sorted(base_dir.iterdir(), key=lambda p: (p.is_file(), p.name))
    entries = [e for e in entries if e.name != ".gitkeep"]  # .gitkeep 숨김 / Hide .gitkeep
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        if entry.is_dir():
            print(f"{indent}{connector}📁 {entry.name}/")
            ext = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, indent + ext, max_depth, _depth + 1)
        else:
            print(f"{indent}{connector}📄 {entry.name}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Grayspot — Storage Initialization")
    print("  (data/ 기준 · download_dataset.py 연동)")
    print("  (Based on data/ · Integrated with download_dataset.py)")
    print("=" * 55)
    cfg = load_config()
    create_storage(cfg)