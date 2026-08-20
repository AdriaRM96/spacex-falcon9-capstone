# SpaceX Falcon 9 First Stage Landing Prediction

Capstone project for the IBM Data Science Professional Certificate (Coursera). The goal is to predict whether the first stage of a SpaceX Falcon 9 rocket will land successfully, since a successful landing (and reuse of the first stage) is the main driver behind SpaceX's ability to offer launches at a fraction of the cost of competitors.

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [01_data_collection_api.ipynb](notebooks/01_data_collection_api.ipynb) | Collects launch data from the SpaceX REST API and performs initial cleaning/wrangling of the raw JSON response. |
| 2 | [02_data_collection_webscraping.ipynb](notebooks/02_data_collection_webscraping.ipynb) | Scrapes historical Falcon 9 launch records from Wikipedia using BeautifulSoup as a complementary data source. |
| 3 | [03_data_wrangling.ipynb](notebooks/03_data_wrangling.ipynb) | Exploratory data analysis and creation of the binary landing outcome label (`Class`) used for supervised training. |
| 4 | [04_eda_sql.ipynb](notebooks/04_eda_sql.ipynb) | Exploratory data analysis using SQL queries (SQLite) to answer questions about launch sites, payload mass, and mission outcomes. |
| 5 | [05_eda_dataviz.ipynb](notebooks/05_eda_dataviz.ipynb) | Visual EDA and feature engineering with `matplotlib`/`seaborn` (flight number, payload, orbit vs. success rate) plus one-hot encoding for modeling. |
| 6 | [06_launch_site_location_folium.ipynb](notebooks/06_launch_site_location_folium.ipynb) | Interactive geographic analysis of launch site locations and their proximity to coastlines, railways, highways, and cities using Folium. |
| 7 | [07_machine_learning_prediction.ipynb](notebooks/07_machine_learning_prediction.ipynb) | Trains and tunes (via `GridSearchCV`) Logistic Regression, SVM, Decision Tree, and KNN classifiers to predict landing success, and compares their performance. |

> **Note on Notebook 1:** at the time this notebook was completed, the public SpaceX API (`api.spacexdata.com`) was returning repeated `525` errors (server-side SSL handshake failure), an outage on the external service's side, not in this code. The notebook's code cells are complete and correct, but the API calls could not be executed end-to-end while the outage lasted. Re-running the notebook once the API is available again will produce the expected outputs.

## Dashboard

An interactive [Plotly Dash](https://dash.plotly.com/) application ([dashboard/spacex-dash-app.py](dashboard/spacex-dash-app.py)) lets you explore launch success by site and payload range.

To run it locally:

```bash
pip install pandas dash
cd dashboard
python spacex-dash-app.py
```

Then open `http://127.0.0.1:8050/` in your browser. The app includes:
- A dropdown to filter by launch site (or view all sites)
- A pie chart of successful launches per site
- A payload mass range slider (0–10,000 kg)
- A scatter plot of payload mass vs. landing outcome, colored by booster version

## Data

The `data/` folder contains the datasets used by the SQL and dashboard notebooks:
- `spacex_launch_dash.csv` — data feeding the Plotly Dash dashboard
- `spacex_launch_geo.csv` — launch site coordinates used in the Folium map
- `my_data1.db` — SQLite database used in the SQL EDA notebook

## Key Results

- **Overall first-stage landing success rate:** 66.7% across the analyzed launches.
- **Best performing model (cross-validation):** Decision Tree Classifier, with **87.5% accuracy** on 10-fold cross-validation — the highest among Logistic Regression, SVM, and KNN, all of which scored between 84.6% and 84.8%.
- **Test-set accuracy:** all four models tied at 83.3% on the (small, 18-sample) held-out test set, which is expected given how few samples that split contains — cross-validation accuracy is the more reliable comparison metric here.
- **Geographic finding:** all launch sites sit within ~1 km of the coastline, supporting safe over-water trajectories, while keeping distance from populated areas.
- **Orbit finding:** payloads to LEO, ISS, and Polar orbits show higher landing success rates than GTO missions.

## Presentation

The final slide deck summarizing the full analysis is available at [presentation/SpaceX_Capstone_Presentation.pptx](presentation/SpaceX_Capstone_Presentation.pptx).
