#!/usr/bin/env python3
"""Refresh the cached Google Scholar citation count shown on the homepage."""

from __future__ import annotations

import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHOLAR_ID = os.environ.get("SCHOLAR_ID", "KJSB8EkAAAAJ")
PROFILE_URL = (
    "https://scholar.google.com/citations"
    f"?hl=en&user={SCHOLAR_ID}"
)
BIO_PATH = Path(__file__).resolve().parents[1] / "src/content/bio.md"
CITATION_PATTERN = re.compile(
    r'(<span data-scholar-citations>)\d+(</span>)'
)
SCHOLAR_PATTERN = re.compile(
    r'<td class="gsc_rsb_sc1"><a[^>]*>Citations</a></td>'
    r'<td class="gsc_rsb_std">([\d,]+)</td>'
)


def fetch_citations() -> int:
    request = urllib.request.Request(
        PROFILE_URL,
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        },
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
            match = SCHOLAR_PATTERN.search(html)
            if not match:
                raise RuntimeError(
                    "Google Scholar returned a page without citation statistics"
                )
            return int(match.group(1).replace(",", ""))
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"Unable to fetch Google Scholar citations: {last_error}")


def main() -> int:
    source = BIO_PATH.read_text(encoding="utf-8")
    current_match = CITATION_PATTERN.search(source)
    if not current_match:
        raise RuntimeError(
            f"Citation marker not found in {BIO_PATH}"
        )

    current = int(
        re.search(r"\d+", current_match.group(0)).group(0)  # type: ignore[union-attr]
    )
    latest = fetch_citations()

    if latest < current:
        print(
            f"Keeping cached citation count {current}; "
            f"fetched value unexpectedly decreased to {latest}."
        )
        return 0
    if latest == current:
        print(f"Google Scholar citation count is already {current}.")
        return 0

    updated = CITATION_PATTERN.sub(
        rf"\g<1>{latest}\g<2>",
        source,
        count=1,
    )
    BIO_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated Google Scholar citations: {current} -> {latest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"warning: {error}", file=sys.stderr)
        raise SystemExit(1)
