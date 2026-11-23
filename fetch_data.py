"""Collect Earth encounters from NASA NeoWs in seven-day windows."""
import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from dataset import COLUMNS, ROOT, one_per_asteroid, output_path

FEED = "https://api.nasa.gov/neo/rest/v1/feed"


def windows(start, end):
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    while start <= end:
        last = min(start + timedelta(days=6), end)
        yield start, last
        start = last + timedelta(days=1)


def fetch_window(start, end, key):
    params = urlencode({"start_date": start, "end_date": end, "api_key": key})
    try:
        with urlopen(f"{FEED}?{params}", timeout=45) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 429:
            raise ValueError("NASA rate limit reached. Retry later or use your own NASA_API_KEY.") from None
        raise ValueError(f"NASA returned HTTP {error.code}. Check the dates and API key.") from None
    except (URLError, TimeoutError):
        raise ValueError("Could not reach NASA. Retry later or train on the included historical CSV.") from None


def earth_rows(payload, first, last):
    """A feed object can list multiple approaches. Match the body AND date."""
    if not isinstance(payload, dict) or not isinstance(payload.get("near_earth_objects"), dict):
        raise ValueError("NASA returned an unexpected feed format.")
    by_day = payload["near_earth_objects"]
    for day, objects in by_day.items():
        if not first <= date.fromisoformat(day) <= last:
            continue
        for obj in objects:
            for approach in obj["close_approach_data"]:
                if approach["orbiting_body"] != "Earth" or approach["close_approach_date"] != day:
                    continue
                flag = obj["is_potentially_hazardous_asteroid"]
                if type(flag) is not bool:
                    raise ValueError("NASA returned a missing or invalid PHA flag.")
                yield {
                    "asteroid_id": obj["id"],
                    "approach_date": day,
                    "absolute_magnitude": obj["absolute_magnitude_h"],
                    "speed_km_s": approach["relative_velocity"]["kilometers_per_second"],
                    "miss_distance_au": approach["miss_distance"]["astronomical"],
                    "pha": int(flag),
                }


def collect(start, end, output):
    key = os.environ.get("NASA_API_KEY") or "DEMO_KEY"
    rows = []
    for first, last in windows(start, end):
        print(f"Fetching {first} through {last}")
        rows.extend(earth_rows(fetch_window(first, last, key), first, last))
    data = one_per_asteroid(pd.DataFrame(rows, columns=COLUMNS))
    # Keep any existing CSV intact if the download or validation fails.
    with output_path(output) as temporary:
        data.to_csv(temporary, index=False)
    print(f"Saved {len(data)} unique asteroids ({data['pha'].sum()} flagged) to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "feed.csv")
    args = parser.parse_args()
    try:
        collect(args.start, args.end, args.out)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
