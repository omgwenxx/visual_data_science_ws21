import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import components as c
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from ipywidgets import widgets
import pandas as pd
import numpy as np

final = pd.read_csv("./data/final/final_merge_region.csv")
countries = pd.read_csv("./helper_data/slim-3.csv")
countries.rename(columns={'alpha-3': 'country', 'name': 'country_name'}, inplace=True)
df = pd.merge(final, countries, on="country")


############### INIT PLOTS #####################
def generate_scatter_plot(id, model_set, color):
    df = get_data(model_set)
    fig = px.scatter(df, x="y_true", y="y_pred", color_discrete_sequence=[color],
                     labels={'y_true': 'Ground Truth', 'y_pred': 'Predicted Happiness Score'})
    fig.update_xaxes(range=[0, 10])
    fig.update_yaxes(range=[0, 10])
    fig.add_shape(type="line", line=dict(
        color="#d3d3d3",
        width=2,
    ),
                  y0=0,
                  y1=10,
                  x1=10,
                  x0=0)
    fig.update_layout(transition_duration=800)
    return dcc.Graph(
        id=id,
        figure=fig
    )


def generate_world_map():
    fig = px.choropleth(df, locations="country", width=1000, height=800,
                        color="score", color_continuous_scale=px.colors.diverging.RdBu[::-1],
                        hover_name="country_name",
                        range_color=(2.5, 8),
                        hover_data=["family", 'freedom', 'generosity'],
                        locationmode="ISO-3")
    fig.update_geos(fitbounds="locations")
    return fig


def generate_parallel_coordinates():
    final['region_code'] = final['region'].map({'Americas': 1,
                                                'Oceania': 2,
                                                'Asia': 3,
                                                'Europe': 4,
                                                'Africa': 5,
                                                })
    fig = go.Figure(data=
                    go.Parcoords(line=dict(color=final['region_code'],
                                           colorscale=[[0, '#636EFA'],
                                                       [0.2, '#FFA15A'],
                                                       [0.4, '#D6D9FE'],
                                                       [0.6, '#EF553B'],
                                                       [0.8, '#00CC96']]),
                                 dimensions=list([
                                     dict(label='Generosity', values=final["generosity"]),
                                     dict(label='Freedome', values=final["freedom"]),
                                     dict(label='Family', values=final["family"]),
                                     dict(label='Health', values=final["health"]),
                                     dict(label='Trust', values=final["trust"]),
                                     dict(label='Happiness', values=final["score"]),
                                 ])))

    return dcc.Graph(
        id="parallel_coordinates",
        figure=fig
    )


############### UTILS #####################
def get_data(value):
    switch = {
        "lr1": pd.read_csv("./data/model_results_lr.csv"),
        "lr2": pd.read_csv("./data/model_results_lr_nocountries.csv"),
        "lr3": pd.read_csv("./data/model_results_lr_nocountries_withvariancestuff.csv"),
        "ada1": pd.read_csv("./data/model_results_ada.csv"),
        "ada2": pd.read_csv("./data/model_results_ada_nocountries.csv"),
        "ada3": pd.read_csv("./data/model_results_ada_nocountries_withvariancestuff.csv"),
        "rf1": pd.read_csv("./data/model_results_rf.csv"),
        "rf2": pd.read_csv("./data/model_results_rf_nocountries.csv"),
        "rf3": pd.read_csv("./data/model_results_rf_nocountries_withvariancestuff.csv")
    }

    return switch.get(value, "Invalid input")


def generate_result_text(model_set, id = None):
    df = get_data(model_set)
    mse_string = f"MSE: {round(mse(df.y_true, df.y_pred), 2)}"
    r2_string = f"R2: {round(r2_score(df.y_true, df.y_pred), 2)}"
    html.P(['Why no', html.Br(), 'linebreak?'])
    if id:
        return html.Div(html.P([mse_string, html.Br(), r2_string]), id=id, className="font-weight-bold")
    else:
        return html.P([mse_string, html.Br(), r2_string])


############### DASHBOARD COMPONENTS #####################
# Input Model Setting
def generate_model_input(header, id, value):
    return dbc.Form([
        html.H4(header),
        dcc.Dropdown(id=id,
                     options=[{"label": "Linear Regression (Setting 1)", "value": "lr1"},
                              {"label": "AdaBoost (Setting 1)", "value": "ada1"},
                              {"label": "Random Forest (Setting 1)", "value": "rf1"},
                              {"label": "Linear Regression (Setting 2)", "value": "lr2"},
                              {"label": "AdaBoost (Setting 2)", "value": "ada2"},
                              {"label": "Random Forest (Setting 2)", "value": "rf2"},
                              {"label": "Linear Regression (Setting 3)", "value": "lr3"},
                              {"label": "AdaBoost (Setting 3)", "value": "ada3"},
                              {"label": "Random Forest (Setting 3)", "value": "rf3"},
                              ],
                     value=value)])


def preprocess_data():
    final = pd.read_csv("./data/final/final_merge_region.csv")
    # remove countries that are missing score
    final = final[~final['score'].isnull()]

    # for visualizing it makes only sense to keep columns that have less than
    col_keep = final.columns[final.isnull().mean() < 0.72]  # col with less than 28% missing

    final = final[col_keep]

    countries = pd.read_csv("./helper_data/slim-3.csv")
    countries_name = countries.rename(columns={'alpha-3': 'country', 'name': 'country_name'})
    df = pd.merge(final, countries_name, on="country")
    df['region_code'] = df['region'].map({'Asia': 0.10,
                                          'Europe': 0.30,
                                          'Africa': 0.50,
                                          'Americas': 0.70,
                                          'Oceania': 0.90,
                                          })
    df['sub_region_code'] = df['sub-region'].map({
    'Southern Asia':0.01, #1
    'Southern Europe':0.1, #2
    'Northern Africa':0.2, #3
    'Sub-Saharan Africa':0.25, #4
    'Latin America and the Caribbean':0.3, #5
    'Western Asia':0.37, #6
    'Australia and New Zealand':0.45, #7
    'Western Europe':0.5, #8
    'Eastern Europe':0.6, #9
    'South-eastern Asia':0.69, #10
    'Northern America':0.76, #11
    'Eastern Asia':0.8, #12
    'Northern Europe':0.9, #13
    'Central Asia':0.986 #14
})
    df['dummyKey'] = range(len(df))
    return df


def generate_scatter_input(label, id, value):
    return dbc.Form([
        html.Label(label),
        dcc.Dropdown(id=id,
                     options=[{"label": "Family", "value": "family"},
                              {"label": "Freedom", "value": "freedom"},
                              {"label": "Economy (GDP per Capita)", "value": "gdp_hr"},
                              {"label": "Trust (Government Corruption)", "value": "trust"},
                              {"label": "Generosity", "value": "generosity"},
                              {"label": "Happiness Score", "value": "score"}
                              ],
                     value=value)])


def generate_scatter_color_input():
    return dbc.Form([
        html.Label("Color Coding"),
        dcc.Dropdown(id="scatter_color",
                     options=[{"label": "Continent", "value": "region"},
                              {"label": "Sub-Continent", "value": "sub-region"}],
                     value="region")])
