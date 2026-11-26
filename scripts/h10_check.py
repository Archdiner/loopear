import json
import resource
from pathlib import Path

from loopear.slots import full_report


def main() -> None:
    report = full_report(seconds=10.0)
    report["peak_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = Path("results/h10_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
