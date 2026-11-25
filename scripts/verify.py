import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", "build", "dist", ".pytest_cache"}


def _iter_repo_files(suffix: str) -> list[Path]:
    found: list[Path] = []
    for path in sorted(ROOT.rglob(f"*{suffix}")):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel_parts):
            continue
        found.append(path)
    return found


def check_zero_comments() -> list[str]:
    violations: list[str] = []
    for py_path in _iter_repo_files(".py"):
        with py_path.open("r", encoding="utf-8") as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type == tokenize.COMMENT:
                    rel = py_path.relative_to(ROOT)
                    violations.append(f"{rel}:{tok.start[0]}:{tok.start[1]}: {tok.string}")
    return violations


def check_readme_only() -> list[str]:
    md_files = [str(p.relative_to(ROOT)) for p in _iter_repo_files(".md")]
    if md_files != ["README.md"]:
        return [f"Expected only ['README.md'], found {md_files}"]
    return []


def run_cmd(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=ROOT)
    return int(proc.returncode)


def main() -> int:
    comment_errors = check_zero_comments()
    if comment_errors:
        print("ZERO-COMMENT CHECK FAILED:")
        for err in comment_errors:
            print(f"  {err}")
        return 1
    py_count = len(_iter_repo_files(".py"))
    print(f"ZERO-COMMENT CHECK PASSED ({py_count} .py files)")

    md_errors = check_readme_only()
    if md_errors:
        print("README-ONLY CHECK FAILED:")
        for err in md_errors:
            print(f"  {err}")
        return 1
    print("README-ONLY CHECK PASSED (only README.md)")

    pyflakes_rc = run_cmd(
        [sys.executable, "-m", "pyflakes", "loopear", "scripts", "tests"]
    )
    if pyflakes_rc != 0:
        print("PYFLAKES CHECK FAILED")
        return pyflakes_rc
    print("PYFLAKES CHECK PASSED")

    pytest_rc = run_cmd([sys.executable, "-m", "pytest", "-q", "tests"])
    if pytest_rc != 0:
        print("PYTEST FAILED")
        return pytest_rc
    print("ALL VERIFICATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
