"""Wikipedia launch-table scraping utilities.

Backs notebook 02 (its original scrape logic stays inline there, since it
worked and was already verified) and notebook 08 (the dataset refresh),
which needs the same parsing logic applied across several live year-range
pages instead of one frozen snapshot.
"""

from __future__ import annotations

import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def extract_column_from_header(row: Tag) -> str | None:
    """Extract a clean column name from a `<th>` element, dropping refs/line breaks."""
    if row.br:
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()

    column_name = " ".join(row.contents)
    if not column_name.strip().isdigit():
        return column_name.strip()
    return None


def parse_date_time(table_cells: Tag) -> list[str]:
    """Return [date, time] strings from a launch table's date/time cell."""
    return [text.strip() for text in table_cells.strings][0:2]


def parse_booster_version(table_cells: Tag) -> str:
    """Return the booster version string from a launch table cell."""
    return "".join(
        text for i, text in enumerate(table_cells.strings) if i % 2 == 0
    )[:-1] if list(table_cells.strings) else ""


def parse_landing_status(table_cells: Tag) -> str:
    """Return the booster landing outcome string from a launch table cell."""
    return next(iter(table_cells.strings), "")


def parse_payload_mass(table_cells: Tag) -> float:
    """Return the payload mass in kg (float) from a launch table cell, or 0.0 if blank."""
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if not mass or "kg" not in mass:
        return 0.0
    numeric = mass[: mass.find("kg")].replace(",", "").strip()
    try:
        return float(numeric)
    except ValueError:
        return 0.0


def find_launch_tables(soup: BeautifulSoup) -> list[Tag]:
    """Find the actual launch-record tables on a page.

    Matches any `<table>` that has both `plainrowheaders` and `collapsible`
    classes, rather than an exact class-string match. Wikipedia has since
    added a `sticky-header` class to these tables (not present when notebook
    02's snapshot was taken), which would silently break an exact-match
    selector -- checking for the two classes that actually distinguish a
    launch-record table from a generic wikitable is robust to that kind of
    drift.
    """
    tables = []
    for table in soup.find_all("table", class_="wikitable"):
        classes = table.get("class", [])
        if "plainrowheaders" in classes and "collapsible" in classes:
            tables.append(table)
    return tables


def scrape_launch_page(url: str, headers: dict | None = None, timeout: int = 20) -> pd.DataFrame:
    """Scrape all launch-record tables on a Wikipedia launches page.

    Only tables of past launches with a known outcome are parsed, not
    "upcoming/planned" manifest tables (which have fewer columns and no
    outcome yet). Tables are told apart by checking for a `Launch outcome`
    header -- not `Booster landing`, even though that's the column we
    actually want: its `<th>` is entirely wrapped in an `<a>` link
    ("Falcon 9 first-stage landing tests"), so
    `extract_column_from_header`'s `row.a.extract()` step strips the link
    *and* its text, silently losing that header name. This isn't a new
    issue -- notebook 02's `column_names` output has the same gap and works
    around it by adding `launch_dict['Booster landing']` by hand rather than
    relying on the auto-extracted header list.

    Returns a DataFrame with columns: FlightNumber, Date, Time,
    BoosterVersion, LaunchSite, Payload, PayloadMass, Orbit, Customer,
    Outcome, BoosterLanding.
    """
    response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows_out: list[dict] = []
    for table in find_launch_tables(soup):
        header_names = [
            name
            for th in table.find_all("th")
            if (name := extract_column_from_header(th)) is not None
        ]
        if "Launch outcome" not in header_names:
            continue

        for tr in table.find_all("tr"):
            if not (tr.th and tr.th.string and tr.th.string.strip().isdigit()):
                continue
            flight_number = tr.th.string.strip()
            cells = tr.find_all("td")
            if len(cells) < 9:
                continue

            date_time = parse_date_time(cells[0])
            booster_version = parse_booster_version(cells[1]) or (cells[1].a.string if cells[1].a else None)

            rows_out.append(
                {
                    "FlightNumber": int(flight_number),
                    "Date": date_time[0].strip(",") if date_time else None,
                    "Time": date_time[1] if len(date_time) > 1 else None,
                    "BoosterVersion": booster_version,
                    "LaunchSite": cells[2].a.string if cells[2].a else None,
                    "Payload": cells[3].a.string if cells[3].a else None,
                    "PayloadMass": parse_payload_mass(cells[4]),
                    "Orbit": cells[5].a.string if cells[5].a else None,
                    "Customer": cells[6].a.string if cells[6].a else None,
                    "Outcome": next(iter(cells[7].strings), None),
                    "BoosterLanding": parse_landing_status(cells[8]),
                }
            )

    return pd.DataFrame(rows_out)


def scrape_multiple_pages(urls: list[str], headers: dict | None = None) -> pd.DataFrame:
    """Scrape and concatenate launch records from several Wikipedia pages.

    Each page is scraped independently and any that raise (network error,
    unexpected structure) are skipped with a printed warning rather than
    aborting the whole run -- partial, honestly-labeled data beats a hard
    failure when combining several live, independently-maintained pages.
    """
    frames = []
    for url in urls:
        try:
            df = scrape_launch_page(url, headers=headers)
            if df.empty:
                print(f"Warning: no launch-record tables found on {url}")
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            print(f"Warning: failed to scrape {url} ({exc})")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
