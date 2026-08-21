# Import required libraries
from pathlib import Path

import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the SpaceX launch data into a pandas dataframe
# Resolved relative to this file (not the current working directory), so the
# app runs the same whether launched from dashboard/, the repo root, or by
# gunicorn on Render.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spacex_launch_dash.csv"
spacex_df = pd.read_csv(DATA_PATH)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)
# Exposes the underlying Flask app so a production WSGI server (gunicorn) can
# serve it -- app.run()'s built-in server below is fine for local development
# but isn't meant for production traffic.
server = app.server

# Create an app layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),

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
    html.Br(),

    # TASK 2: pie chart showing success counts, updated by the dropdown callback
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    html.P("Payload range (Kg):"),
    # TASK 3: range slider from 0 to 10000 kg in steps of 1000 kg
    dcc.RangeSlider(id='payload-slider',
                     min=0,
                     max=10000,
                     step=1000,
                     marks={i: str(i) for i in range(0, 10001, 2000)},
                     value=[min_payload, max_payload]),

    # TASK 4: scatter chart showing payload vs. outcome, updated by dropdown + slider
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
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
                     title='Total Success Launches by Site')
        return fig
    else:
        # For one site, show success (1) vs failure (0) counts within that site only
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        outcome_counts = filtered_df['class'].value_counts().reset_index()
        outcome_counts.columns = ['class', 'count']
        fig = px.pie(outcome_counts, values='count',
                     names='class',
                     title=f'Total Success Launches for site {entered_site}')
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
    return fig


# Run the app
if __name__ == '__main__':
    app.run(port=8050)
