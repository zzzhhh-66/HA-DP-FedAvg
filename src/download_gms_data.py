"""Download Give Me Some Credit dataset via kagglehub."""

import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "give_me_some_credit"


def main():
    path = kagglehub.competition_download("GiveMeSomeCredit")
    print("Path to competition files:", path)

    source = Path(path)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for csv_name in ("cs-training.csv", "cs-test.csv", "sampleEntry.csv"):
        src = source / csv_name
        if src.exists():
            dst = TARGET_DIR / csv_name
            shutil.copy2(src, dst)
            print(f"Copied {csv_name} -> {dst}")
        else:
            print(f"Warning: {csv_name} not found in {source}")


if __name__ == "__main__":
    main()
