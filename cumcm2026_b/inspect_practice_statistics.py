"""Inspect the official practice-statistics queue without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    database = args.database.resolve()
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    result = {}
    for table in tables:
        quoted = table.replace('"', '""')
        result[table] = {
            "columns": [
                {"name": row[1], "type": row[2]}
                for row in connection.execute(f'PRAGMA table_info("{quoted}")')
            ],
            "rows": list(
                connection.execute(
                    f'SELECT * FROM "{quoted}" ORDER BY rowid DESC LIMIT ?',
                    (args.rows,),
                )
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
