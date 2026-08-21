# Raw data snapshots

These are frozen, point-in-time copies of the external sources the pipeline pulls from, so the notebooks can be re-run without depending on those services staying online or unchanged.

- **`spacex_api_launches_raw.json`** — the exact JSON response returned by the SpaceX launches endpoint (fetched via the IBM-hosted static mirror of `GET /v4/launches/past`, used in [`notebooks/01_data_collection_api.ipynb`](../../notebooks/01_data_collection_api.ipynb) as `static_json_url`). 107 launch records, saved with `requests.get(url).json()`.
- **`falcon9_wikipedia_launches_raw.html`** — the raw HTML of the Wikipedia snapshot used in [`notebooks/02_data_collection_webscraping.ipynb`](../../notebooks/02_data_collection_webscraping.ipynb) (`List of Falcon 9 and Falcon Heavy launches`, revision `oldid=1027686922`, fetched 2021-06-09). Saved with `requests.get(url, headers=headers).text`.

Notebook 01 falls back to `spacex_api_launches_raw.json` automatically if the live SpaceX API is unreachable (it has had recurring outages — see the note in that notebook). Notebook 02's live source (Wikipedia) has been reliable so far, so it still fetches live by default; the HTML snapshot here exists purely as a reproducibility backstop if that ever changes.
