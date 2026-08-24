# Import required libraries
from pathlib import Path

import joblib
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from flask import jsonify, request
import plotly.express as px

from spacex_capstone.inference import VALID_LAUNCH_SITES, VALID_ORBITS, predict_landing

# Read the SpaceX launch data into a pandas dataframe
# Resolved relative to this file (not the current working directory), so the
# app runs the same whether launched from dashboard/, the repo root, or by
# gunicorn on Render.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spacex_launch_dash.csv"
spacex_df = pd.read_csv(DATA_PATH)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Static mission-control style stats shown as telemetry badges at the top of
# the page -- these summarize the whole dataset once at load time, unrelated
# to the interactive dropdown/slider filters below.
TOTAL_LAUNCHES = len(spacex_df)
OVERALL_SUCCESS_RATE = spacex_df['class'].mean() * 100
LAUNCH_SITES = spacex_df['Launch Site'].nunique()

# The trained model artifact is generated at build time by
# train_and_export.py (see render.yaml's buildCommand) rather than committed
# to the repo -- see that script's docstring for why. Locally, run it once
# yourself before starting the app:
#   python -m spacex_capstone.train_and_export dashboard/model_artifact.joblib
MODEL_ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifact.joblib"
try:
    _artifact = joblib.load(MODEL_ARTIFACT_PATH)
except FileNotFoundError:
    _artifact = None
    print(
        f"No model artifact at {MODEL_ARTIFACT_PATH} -- the Predict tab and /predict route "
        "will report an error until you run: python -m spacex_capstone.train_and_export "
        "dashboard/model_artifact.joblib"
    )

# Google Fonts: Rajdhani (condensed geometric sans for headings/body) and
# JetBrains Mono (for telemetry-style counters), loaded the same way any
# external stylesheet is in Dash.
FONT_STYLESHEET = "https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap"

# A dark Plotly theme so charts blend with the page instead of showing up as
# bright white rectangles against the mission-control background.
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111214",
    plot_bgcolor="#111214",
    font=dict(family="Rajdhani, sans-serif", color="#e8e8e8", size=14),
    title_font=dict(family="Rajdhani, sans-serif", size=18),
)

# Create a dash application
app = dash.Dash(__name__, external_stylesheets=[FONT_STYLESHEET])
app.title = "SpaceX Launch Analytics"
# Exposes the underlying Flask app so a production WSGI server (gunicorn) can
# serve it -- app.run()'s built-in server below is fine for local development
# but isn't meant for production traffic.
server = app.server


def telemetry_badge(label: str, value: str, status: str = "") -> html.Div:
    """A small mission-control-style stat badge (label on top, value below)."""
    classes = "telemetry-badge" + (f" status-{status}" if status else "")
    return html.Div(
        className=classes,
        children=[
            html.Span(label, className="label"),
            html.Span(value, className="value"),
        ],
    )


# Create an app layout
app.layout = html.Div(className="mission-control", children=[

    html.Div(className="header", children=[
        html.H1("SpaceX Launch Analytics"),
        html.Div("FALCON 9 FIRST STAGE LANDING PREDICTION SYSTEM", className="subtitle"),
    ]),

    html.Div(className="telemetry-row", children=[
        telemetry_badge("Total Launches", str(TOTAL_LAUNCHES)),
        telemetry_badge(
            "Landing Success Rate",
            f"{OVERALL_SUCCESS_RATE:.1f}%",
            status="good" if OVERALL_SUCCESS_RATE >= 50 else "warn",
        ),
        telemetry_badge("Launch Sites Tracked", str(LAUNCH_SITES)),
    ]),

    dcc.Tabs(id='main-tabs', value='tab-analytics', className='mc-tabs', children=[
        dcc.Tab(label='ANALYTICS', value='tab-analytics', className='tab', selected_className='tab--selected', children=[
            html.Div(className="tab-content", children=[
                html.Div(className="controls-panel", children=[
                    html.Div("SELECT LAUNCH SITE", className="section-label"),
                    # TASK 1: dropdown with an "ALL" option + one option per launch site,
                    # placeholder text, and search enabled (searchable=True is the default)
                    dcc.Dropdown(id='site-dropdown',
                                 options=[
                                     {'label': 'All Sites', 'value': 'ALL'},
                                     {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                     {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                                     {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                     {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                 ],
                                 value='ALL',
                                 placeholder="Select a Launch Site here",
                                 searchable=True
                                 ),

                    html.Div("PAYLOAD MASS RANGE (KG)", className="section-label"),
                    # TASK 3: range slider from 0 to 10000 kg in steps of 1000 kg
                    dcc.RangeSlider(id='payload-slider',
                                     min=0,
                                     max=10000,
                                     step=1000,
                                     marks={i: str(i) for i in range(0, 10001, 2000)},
                                     value=[min_payload, max_payload]),
                ]),

                # TASK 2: pie chart showing success counts, updated by the dropdown callback
                html.Div(className="chart-panel", children=[dcc.Graph(id='success-pie-chart')]),

                # TASK 4: scatter chart showing payload vs. outcome, updated by dropdown + slider
                html.Div(className="chart-panel", children=[dcc.Graph(id='success-payload-scatter-chart')]),
            ]),
        ]),
        dcc.Tab(label='PREDICT', value='tab-predict', className='tab', selected_className='tab--selected', children=[
            html.Div(className="tab-content", children=[
                html.Div(className="controls-panel", children=[
                    html.Div("PREDICT A LAUNCH", className="section-label"),
                    html.P(
                        "Get a real landing-probability prediction from the trained model -- not a lookup "
                        "over past launches, but the model's estimate for a hypothetical mission.",
                        style={"color": "#8a8f98", "fontSize": "0.85rem", "marginTop": "-4px"},
                    ),
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "flex-end"}, children=[
                        html.Div(children=[
                            html.Label("Payload mass (kg)", style={"fontSize": "0.8rem", "color": "#8a8f98"}),
                            dcc.Input(id='predict-payload-mass', type='number', value=4000, min=0, max=20000, step=100),
                        ]),
                        html.Div(children=[
                            html.Label("Orbit", style={"fontSize": "0.8rem", "color": "#8a8f98"}),
                            dcc.Dropdown(
                                id='predict-orbit',
                                options=[{'label': o, 'value': o} for o in VALID_ORBITS],
                                value='LEO',
                                style={"width": "160px"},
                            ),
                        ]),
                        html.Div(children=[
                            html.Label("Launch site", style={"fontSize": "0.8rem", "color": "#8a8f98"}),
                            dcc.Dropdown(
                                id='predict-launch-site',
                                options=[{'label': s, 'value': s} for s in VALID_LAUNCH_SITES],
                                value=VALID_LAUNCH_SITES[0],
                                style={"width": "220px"},
                            ),
                        ]),
                        html.Button("PREDICT", id='predict-button', n_clicks=0, className='predict-button'),
                    ]),
                    html.Div(id='predict-result', className="telemetry-row", style={"marginTop": "20px"}),
                ]),
            ]),
        ]),
    ]),

    html.Div(
        "Independent data analysis project. Not affiliated with or endorsed by SpaceX.",
        className="footer",
    ),
])


# TASK 2: callback for the success pie chart
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # For all sites, count how many successful launches (class=1) each site had
        fig = px.pie(spacex_df, values='class',
                     names='Launch Site',
                     title='Total Success Launches by Site',
                     color_discrete_sequence=['#e5312f', '#1a5fb4', '#2ec27e', '#f5c211'])
    else:
        # For one site, show success (1) vs failure (0) counts within that site only
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        outcome_counts = filtered_df['class'].value_counts().reset_index()
        outcome_counts.columns = ['class', 'count']
        fig = px.pie(outcome_counts, values='count',
                     names='class',
                     title=f'Total Success Launches for site {entered_site}',
                     color_discrete_sequence=['#2ec27e', '#e5312f'])
    fig.update_layout(**CHART_LAYOUT)
    return fig


# TASK 4: callback for the payload vs. outcome scatter chart
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def get_scatter_chart(entered_site, payload_range):
    low, high = payload_range
    # Keep only launches whose payload mass falls inside the slider's selected range
    mask = (spacex_df['Payload Mass (kg)'] >= low) & (spacex_df['Payload Mass (kg)'] <= high)
    filtered_df = spacex_df[mask]

    if entered_site == 'ALL':
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                          color='Booster Version Category',
                          title='Correlation between Payload and Success for All Sites')
    else:
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                          color='Booster Version Category',
                          title=f'Correlation between Payload and Success for site {entered_site}')
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _run_prediction(payload_mass, orbit, launch_site):
    """Shared by the Dash callback and the /predict route so both use the
    exact same inference call -- see src/spacex_capstone/inference.py."""
    if _artifact is None:
        raise RuntimeError(
            "Model artifact not found. Run: python -m spacex_capstone.train_and_export "
            "dashboard/model_artifact.joblib"
        )
    return predict_landing(
        payload_mass=float(payload_mass),
        orbit=orbit,
        launch_site=launch_site,
        model=_artifact["model"],
        scaler=_artifact["scaler"],
        feature_names=_artifact["feature_names"],
        baseline=_artifact["baseline"],
    )


# Dash callback: powers the "PREDICT" button in the dashboard's Predict panel.
@app.callback(
    Output(component_id='predict-result', component_property='children'),
    Input(component_id='predict-button', component_property='n_clicks'),
    [State(component_id='predict-payload-mass', component_property='value'),
     State(component_id='predict-orbit', component_property='value'),
     State(component_id='predict-launch-site', component_property='value')],
    prevent_initial_call=True,
)
def on_predict_click(n_clicks, payload_mass, orbit, launch_site):
    try:
        result = _run_prediction(payload_mass, orbit, launch_site)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return html.Div(str(exc), className="telemetry-badge status-warn")

    return [
        telemetry_badge(
            "Landing Success Probability",
            f"{result.p_success:.1%}",
            status="good" if result.p_success >= 0.5 else "warn",
        ),
        telemetry_badge("Expected Launch Cost", f"${result.expected_cost_usd / 1_000_000:.1f}M"),
    ]


# Plain Flask route on the same server Dash uses -- see the project README
# for why this isn't a separate FastAPI service. Example:
#   curl "http://127.0.0.1:8050/predict?payload_mass=4000&orbit=ISS&launch_site=CCAFS%20SLC%2040"
@server.route('/predict', methods=['GET'])
def predict_route():
    try:
        payload_mass = float(request.args.get('payload_mass', 4000))
        orbit = request.args.get('orbit', 'LEO')
        launch_site = request.args.get('launch_site', VALID_LAUNCH_SITES[0])
        result = _run_prediction(payload_mass, orbit, launch_site)
    except Exception as exc:  # noqa: BLE001 - reported as a 400, not a 500 traceback
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "payload_mass": payload_mass,
        "orbit": orbit,
        "launch_site": launch_site,
        "p_success": result.p_success,
        "expected_cost_usd": result.expected_cost_usd,
    })


# Run the app
if __name__ == '__main__':
    app.run(port=8050)
