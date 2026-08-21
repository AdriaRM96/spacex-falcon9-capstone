# SpaceX Falcon 9 — First Stage Landing Prediction

Predicting whether a Falcon 9 first stage will land successfully, and using that prediction to estimate what a SpaceX launch really costs.

## Why this project

SpaceX advertises Falcon 9 launches at around $62M, well under the $165M+ typical of competitors. The gap comes almost entirely from one thing: SpaceX reuses the first stage instead of throwing it away after every flight. If a competitor — or an analyst, investor, or space enthusiast — wants to understand SpaceX's real cost structure, the question that matters isn't "how much does a Falcon 9 launch cost on paper," it's **"how likely is that first stage to come back in one piece?"**

That's the question this project answers end to end: pulling raw launch data from scratch, cleaning it into something usable, exploring what actually drives a successful landing, and building a model that predicts it.

## What's inside

The project follows a full data science workflow, from raw data to a working prediction model:

| Stage | Notebook | What it does |
|---|---|---|
| 1. Collect | [`01_data_collection_api.ipynb`](notebooks/01_data_collection_api.ipynb) | Pulls launch, rocket, payload, and core data from the public SpaceX REST API. |
| 2. Collect | [`02_data_collection_webscraping.ipynb`](notebooks/02_data_collection_webscraping.ipynb) | Scrapes the historical Falcon 9/Falcon Heavy launch table from Wikipedia with BeautifulSoup, as a second, independent data source. |
| 3. Wrangle | [`03_data_wrangling.ipynb`](notebooks/03_data_wrangling.ipynb) | Merges both sources and engineers the binary landing-success label (`Class`) that everything downstream is trained on. |
| 4. Explore | [`04_eda_sql.ipynb`](notebooks/04_eda_sql.ipynb) | SQL-driven exploration (SQLite) — launch sites, payload totals, mission outcomes, booster history. |
| 5. Explore | [`05_eda_dataviz.ipynb`](notebooks/05_eda_dataviz.ipynb) | Visual EDA — how flight number, payload mass, and orbit relate to landing success — plus feature engineering for modeling. |
| 6. Geolocate | [`06_launch_site_location_folium.ipynb`](notebooks/06_launch_site_location_folium.ipynb) | Interactive map of launch sites and their proximity to coastlines, railways, and highways. |
| 7. Predict | [`07_machine_learning_prediction.ipynb`](notebooks/07_machine_learning_prediction.ipynb) | Trains and tunes Logistic Regression, SVM, Decision Tree, and KNN classifiers (`GridSearchCV`) to predict landing outcome. |

> **A note on Notebook 1:** while running this, the public SpaceX API (`api.spacexdata.com`) was intermittently returning `525` errors — an outage on their end, not in this code. The notebook is complete and correct; re-running it once the API is healthy reproduces the pipeline end to end.

## Interactive dashboard

[`dashboard/spacex-dash-app.py`](dashboard/spacex-dash-app.py) is a [Plotly Dash](https://dash.plotly.com/) app for exploring the results interactively — filter by launch site, scan a pie chart of successful launches, or slide through payload mass ranges against landing outcome.

```bash
pip install pandas dash
cd dashboard
python spacex-dash-app.py
```

Then open `http://127.0.0.1:8050/`.

## Data

`data/` holds what powers the SQL notebook, the dashboard, and the Folium map:
- `spacex_launch_dash.csv` — dashboard data
- `spacex_launch_geo.csv` — launch site coordinates
- `my_data1.db` — SQLite database used for the SQL exploration

## What I found

- **Landing success climbed fast.** From effectively 0% between 2010–2014 to 77.8% by 2017 — the clearest signal of SpaceX iterating its recovery process in real time.
- **Site matters.** KSC LC-39A leads with a 76.9% success rate; CCAFS LC-40, used more in the program's early years, trails at 26.9%.
- **Heavier payloads land less reliably.** Across sites and orbits, landing success drops as payload mass increases — physics, not chance.
- **Geography isn't an accident.** Every launch site sits within ~1 km of the coast, keeping failed landings and aborts over water instead of populated land.
- **Decision Tree wins on generalization.** It hit 87.5% cross-validation accuracy, ahead of Logistic Regression, SVM, and KNN (84.6–84.8%). All four tie at 83.3% on the small 18-sample test split — cross-validation is the metric to trust here.

## Presentation

The full analysis is written up as a slide deck: [`presentation/SpaceX_Capstone_Presentation.pptx`](presentation/SpaceX_Capstone_Presentation.pptx).

## Stack

Python · pandas · scikit-learn · SQLite · BeautifulSoup · Folium · Plotly Dash · matplotlib/seaborn

## What's next

- Pull in more recent launches (Block 5, Starship) to see whether the success-rate ceiling has moved.
- Add booster-specific reuse count as a feature — a booster on its 10th flight likely behaves differently than one on its 1st.
- Swap the static dashboard filters for a live-refreshing feed off the SpaceX API.
