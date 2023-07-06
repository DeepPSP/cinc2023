"""
synchronize with the official repo
"""

from pathlib import Path


project_dir = Path(__file__).resolve().parent
official_dir = project_dir / "official_baseline_classifier"

files = [
    "helper_code.py",
    "remove_data.py",
    "remove_labels.py",
    "run_model.py",
    "train_model.py",
    "truncate_data.py",
]


def main():
    for filename in files:
        src = official_dir / filename
        dst = project_dir / filename
        if src.read_text() == dst.read_text():
            continue
        print(
            f"Copying **{src.relative_to(project_dir)}** to **{dst.relative_to(project_dir)}**"
        )
        dst.write_text(src.read_text())


if __name__ == "__main__":
    main()
