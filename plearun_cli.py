#!/usr/bin/env python3
"""
plearun_cli.py - Offline Probabilistic Fact-Checking CLI Entrypoint

Usage:
    python3 plearun_cli.py REPORT RECORDS_DIR > findings.json
"""

import sys
import json
import os

from verifier_engine import verify_report


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plearun_cli.py REPORT RECORDS_DIR", file=sys.stderr)
        sys.exit(1)

    report_path = sys.argv[1]
    records_dir = sys.argv[2]

    if not os.path.exists(report_path):
        print(f"Error: Report file not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(records_dir):
        print(f"Error: Records directory not found: {records_dir}", file=sys.stderr)
        sys.exit(1)

    result = verify_report(report_path, records_dir)

    # Validate output schema strictly
    for item in result.get("findings", []):
        finding = item.get("finding")
        if finding in ("match", "conflict"):
            assert "cite" in item and "quote" in item, "cite and quote required for match/conflict"
            assert isinstance(item["cite"]["file"], str)
            assert isinstance(item["cite"]["row"], int)
            assert isinstance(item["quote"], str)
        else:
            assert "cite" not in item and "quote" not in item, "cite and quote must be omitted for not_settled/not_checkable"

    # Print formatted JSON to stdout
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
