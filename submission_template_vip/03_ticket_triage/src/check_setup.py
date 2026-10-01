"""Sanity-check the environment and data files before we start building models.

Run from anywhere:
    .venv/bin/python submission_template_vip/03_ticket_triage/src/check_setup.py
"""

import importlib
import json
import re
import sys
from pathlib import Path

# This file lives at <root>/submission_template_vip/03_ticket_triage/src/check_setup.py,
# so the project root is 3 folders up. Using __file__ means the script works
# no matter which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = PROJECT_ROOT / "SG_Hackathon_Pack" / "03_ticket_triage" / "artifacts"

PACKAGES = ["pandas", "numpy", "sklearn", "scipy", "joblib", "fastapi", "streamlit"]


def check_python_version():
    """Stop early if we're not on Python 3.12 (project rule: never 3.13)."""
    version = sys.version.split()[0]
    print(f"Python version: {version}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        sys.exit(
            f"❌ Python 3.12 is required, but this is {version}.\n"
            f"   Activate the venv first: source .venv/bin/activate"
        )
    print("✅ Python 3.12\n")


def check_packages():
    """Import each package and print its version, so missing installs show up now."""
    print("Packages:")
    all_ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✅ {name:<10} {version}")
        except ImportError as error:
            print(f"  ❌ {name:<10} NOT INSTALLED ({error})")
            all_ok = False
    print()
    return all_ok


def find_claude_md():
    """CLAUDE.md may be saved as claude.md, so look for either spelling."""
    for name in ["CLAUDE.md", "claude.md"]:
        path = PROJECT_ROOT / name
        if path.exists():
            return path
    sys.exit(f"❌ Could not find CLAUDE.md in {PROJECT_ROOT}")


def get_folders_section(text):
    """Return only the lines between '## Folders' and the next '## ' heading."""
    lines = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Folders"
            continue
        if inside:
            lines.append(line)
    return lines


def paths_from_line(line):
    """Pull every file/folder path mentioned on one bullet line.

    Paths appear in `backticks`, and some lines list bare file names in
    (parentheses) or backticks that belong to the folder mentioned earlier
    on the same line, e.g. "`.../artifacts/` (a.csv, b.json)".
    """
    names = re.findall(r"`([^`]+)`", line)
    for group in re.findall(r"\(([^)]*)\)", line):
        names += [part.strip() for part in group.split(",")]

    paths = []
    base = None  # the most recent folder seen on this line
    for name in names:
        if not name or " " in name:
            continue
        if "/" in name:
            path = PROJECT_ROOT / name
            base = path if name.endswith("/") else path.parent
        elif base is not None:
            path = base / name
        else:
            continue
        # Expand wildcards like "sample_submission*.csv" into the real files.
        # If nothing matches, keep the pattern so it gets reported as missing.
        if "*" in path.name:
            paths += sorted(path.parent.glob(path.name)) or [path]
        else:
            paths.append(path)
    return paths


def count_csv_rows(path):
    """Count data rows (excluding the header) using pandas so quoted newlines are handled."""
    import pandas as pd

    return len(pd.read_csv(path))


def check_files():
    """Print ✅/❌ for every path listed under 'Folders' in CLAUDE.md."""
    claude_md = find_claude_md()
    print(f"Files listed under 'Folders' in {claude_md.name}:")
    all_ok = True
    seen = set()
    for line in get_folders_section(claude_md.read_text()):
        for path in paths_from_line(line):
            if path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(PROJECT_ROOT)
            if not path.exists():
                print(f"  ❌ {relative}  (missing)")
                all_ok = False
            elif path.suffix == ".csv":
                print(f"  ✅ {relative}  ({count_csv_rows(path):,} rows)")
            else:
                print(f"  ✅ {relative}")
    print()
    return all_ok


def show_artifacts():
    """Print the routing cost matrix and capacity limits we'll rely on later."""
    import pandas as pd

    cost_path = ARTIFACTS_DIR / "routing_cost_matrix.csv"
    print(f"=== {cost_path.name} ===")
    if cost_path.exists():
        print(pd.read_csv(cost_path).to_string(index=False))
    else:
        print("❌ missing")
    print()

    capacity_path = ARTIFACTS_DIR / "triage_capacity.json"
    print(f"=== {capacity_path.name} ===")
    if capacity_path.exists():
        print(json.dumps(json.loads(capacity_path.read_text()), indent=2))
    else:
        print("❌ missing")
    print()


def main():
    print(f"Project root: {PROJECT_ROOT}\n")
    check_python_version()
    packages_ok = check_packages()
    files_ok = check_files()
    show_artifacts()

    if packages_ok and files_ok:
        print("✅ Setup looks good.")
    else:
        sys.exit("❌ Setup has problems - see the ❌ lines above.")


if __name__ == "__main__":
    main()
