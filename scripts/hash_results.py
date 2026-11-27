"""Write results/SHA256SUMS: one line per result file, in `sha256sum` format.

Check a tree against it with `sha256sum -c results/SHA256SUMS` from the repository root.
Timing tables (wall-clock measurements) are listed too; they are expected to differ between machines.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"models"}


def main() -> None:
    files = sorted(p for p in (ROOT / "results").rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS" and not SKIP_DIRS & set(p.relative_to(ROOT).parts))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in files]
    (ROOT / "results" / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote results/SHA256SUMS ({len(lines)} files)")


if __name__ == "__main__":
    main()
