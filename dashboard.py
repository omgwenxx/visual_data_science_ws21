# Run this app with `python dashboard.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

import components as c

#################### Declaring Variables and DataFrames #############################
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.title = "VDS Dashboard"
df = c.preprocess_data()
available_indicators = df.columns.unique()
region_color = {
    "Africa": "#00CC96",
    "Europe": "#EF553B",
    "Americas": "#AB63FA",
    "Asia": "#636EFA",
    "Oceania": "#FFA15A"
}
sub_region_color = {
    'Southern Asia': "rgb(99,110,250)",
    'Southern Europe': "#EF553B",
    'Northern Africa': "#16C696",
    'Sub-Saharan Africa': "#FF97FF",
    'Latin America and the Caribbean': "#AB63FA",
    'Western Asia': "#FFA15A",
    'Australia and New Zealand': "#19D3F3",
    'Western Europe': "#FF6692",
    'Eastern Europe': "#B6E880",
    'South-eastern Asia': "#FECB52",
    'Northern America': "#636EFC",
    'Eastern Asia': "#EF553B",
    'Northern Europe': "#00CC96",
    'Central Asia': "#AB63FA"
}

#################### Declaring HTML components #############################
# PLOTS
scatterplot = dcc.Graph(id='crossfilter-indicator-scatter', hoverData={'points': [{'customdata': ['Austria']}]})
donut = dcc.Graph(id='donut')
line_charts = html.Div([dcc.Graph(id='x-time-series'), dcc.Graph(id='y-time-series')])
parallel_coords = dcc.Graph(id="parallel_coordinates")
model_row = dbc.Row([
    dbc.Col(md=6,
            children=[c.generate_model_input("Select Model 1", "model_set1", "lr1"),
                      c.generate_scatter_plot("model1", "lr1", '#FFA15A'),
                      c.generate_result_text("lr1", id='model1_results')]),
    dbc.Col(md=6,
            children=[c.generate_model_input("Select Model 2", "model_set2", "ada2"),
                      c.generate_scatter_plot("model2", "ada2", '#1F77B4'),
                      c.generate_result_text("ada2", id='model2_results')])
])
world_map = dcc.Graph(id='world-plot', figure=c.generate_world_map())

# CONTROLS
dropdown_x = dcc.Dropdown(id='crossfilter-xaxis-column',
                          options=[{'label': i, 'value': i} for i in available_indicators], value='generosity')
dropdown_y = dcc.Dropdown(id='crossfilter-yaxis-column',
                          options=[{'label': i, 'value': i} for i in available_indicators], value='score')
dropdown_year = dcc.Dropdown(id='crossfilter-year--slider',
                             options=[{'label': i, 'value': i} for i in df['year'].unique()], value=df['year'].max())
dropdown_region = dcc.Dropdown(id='crossfilter-region', options=[{'label': "Continent", 'value': "region"},
                                                                 {'label': "Sub-Continent", 'value': "sub-region"}],
                               value="region")
dropdown_country = dcc.Dropdown(id='line-country',
                                options=[{'label': i, 'value': i} for i in df['country_name'].unique()],
                                value="Austria")
scatter_radio_buttons = dbc.RadioItems(id="scatter-radio", options=[
    {'label': 'Histograms', 'value': 'histogram'},
    {'label': 'Boxplot', 'value': 'box'},
    {'label': 'Rug Plot', 'value': 'rug'},
    {'label': 'Violin Plot', 'value': 'violin'},
], value='histogram', className=".form-check-inline")
controls = dbc.Card([
    html.Div([dbc.Label("X variable", className="form-check-label"), dropdown_x], className="mb-3"),
    html.Div([dbc.Label("Y variable", className="form-check-label"), dropdown_y], className="mb-3"),
    html.Div([dbc.Label("Year", className="form-check-label"), dropdown_year], className="mb-3"),
    html.Div([dbc.Label("Marginal Distributions Plot", className="form-check-label"), scatter_radio_buttons],
             className="mb-3"),
    html.Div([dbc.Label("Color Coding", className="form-check-label"), dropdown_region])
], body=True)
controls_map = dbc.Card([
    html.Div([dbc.Label("Variable", className="form-check-label"),
              dcc.Dropdown(id='world-value',
                           options=[{'label': i, 'value': i} for i in available_indicators], value='score')],
             className="mb-3"),
    html.Div([dbc.Label("Year", className="form-check-label"), dcc.Dropdown(id='world-year',
                                                                            options=[{'label': i, 'value': i} for i in
                                                                                     df['year'].unique()],
                                                                            value=df['year'].max())], className="mb-3")
], body=True)

# App Layout
app.layout = dbc.Container([
    html.H1("World Happiness Report + "),
    html.Hr(),
    dbc.Tabs(
        [
            dbc.Tab(label="Data Analysis", tab_id="analysis"),
            dbc.Tab(label="World Map", tab_id="world-map"),
            dbc.Tab(label="Model", tab_id="model"),

        ],
        id="tabs",
        active_tab="analysis",
    ),
    html.Div(id="tab-content", className="p-4")
], fluid=True, className="pt-4")


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        if active_tab == "model":
            return html.Div([
                html.Div([
                    html.H3("Different Settings tried for Modeling"),
                    html.P("Setting 1: Using all the preprocessing steps"),
                    html.P("Setting 2: Using only numerical features and no categorical data (no geolocation informaion)"),
                    html.P("Setting 3: No variance threshold filtering and no categorical data"),
                ])
                ,model_row])
        elif active_tab == "analysis":
            return html.Div([dbc.Row([
                dbc.Col(controls, md=3),
                dbc.Col(scatterplot, md=5, className="pt-5"),
                dbc.Col([html.H4(id="line-header", className="text-center"),
                         line_charts], md=4)]),
                dbc.Row([
                    dbc.Col(donut, md=5),
                    dbc.Col(parallel_coords, md=7)
                ])])
        elif active_tab == "world-map":
            return html.Div([dbc.Row([
                dbc.Col(controls_map, md=2, className="mt-5 pt-5"),
                dbc.Col(world_map, md=10, className="mt-1")
            ])])
    return "No tab selected"


@app.callback(Output('crossfilter-indicator-scatter', 'figure'), [Input('crossfilter-xaxis-column', 'value'),
                                                                  Input('crossfilter-yaxis-column', 'value'),
                                                                  Input('crossfilter-year--slider', 'value'),
                                                                  Input('scatter-radio', 'value'),
                                                                  Input('crossfilter-region', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name, year_value, scatter_value, region):
    dff = df[df['year'] == year_value]
    fig = px.scatter(dff, x=xaxis_column_name, y=yaxis_column_name, color=region, hover_name=dff["country_name"],
                     marginal_y=scatter_value, marginal_x=scatter_value, custom_data=["country_name"])
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig


def create_time_series(dff, title, column):
    fig = px.scatter(dff, x='year', y=column)
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.add_annotation(xanchor='right', yanchor='top',
                       xref='paper', yref='paper', showarrow=False, align='right',
                       text=title)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(Output('x-time-series', 'figure'), Output('line-header', 'children'),
              [Input('crossfilter-indicator-scatter', 'hoverData'),
               Input('crossfilter-xaxis-column', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name):
    title = "Hover over Country"
    try:
        country_name = hoverData['points'][0]['customdata'][0]
        title = country_name
    except:
        print("KeyError customdata but this is no problem")

    dff = df[df['country_name'] == country_name]
    dff = dff[["year", xaxis_column_name]]
    return create_time_series(dff, xaxis_column_name, xaxis_column_name), title


@app.callback(Output('y-time-series', 'figure'), [Input('crossfilter-indicator-scatter', 'hoverData'),
                                                  Input('crossfilter-yaxis-column', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name):
    try:
        dff = df[df['country_name'] == hoverData['points'][0]['customdata'][0]]
    except:
        print("KeyError customdata but this is no problem")

    dff = dff[["year", yaxis_column_name]]
    return create_time_series(dff, yaxis_column_name, yaxis_column_name)


# If data points in Scatterplot are selected via Lasso Tool
# Update Parallel Coordinates with data points
@app.callback([Output('parallel_coordinates', 'figure'),
               Output("donut", "figure")],
              [Input('crossfilter-indicator-scatter', 'selectedData'),
               Input('crossfilter-year--slider', 'value'),
               Input('crossfilter-region', "value")])
def render_donut_parcoords(selectedData, year_value, region_detail):
    df = c.preprocess_data()
    df = df[df['year'] == year_value]
    region_data = df.groupby("region", dropna=False).count()[["country"]].reset_index()
    sub_region_colorscale = [(0, "#636EFA"), (0.07, "#636EFA"), #1
                             (0.07, "#EF553B"), (0.14, "#EF553B"), #2
                             (0.14, "#16C696"), (0.21, "#16C696"), #3
                             (0.21, "#FF97FF"), (0.28, "#FF97FF"), #4
                             (0.28, "#AB63FA"), (0.35, "#AB63FA"), #5
                             (0.35, "#FFA15A"), (0.42, "#FFA15A"), #6
                             (0.42, "#19D3F3"), (0.49, "#19D3F3"), #7
                             (0.49, "#FF6692"), (0.56, "#FF6692"), #8
                             (0.56, "#B6E880"), (0.63, "#B6E880"), #9
                             (0.63, '#FECB52'), (0.70, '#FECB52'), #10
                             (0.70, '#636EFC'), (0.77, '#636EFC'), #11
                             (0.77, '#EF553B'), (0.84, '#EF553B'), #12
                             (0.84, '#00CC96'), (0.91, '#00CC96'), #13
                             (0.91, '#AB63FA'), (1.0, '#AB63FA')] #14
    region_colorscale = [(0, '#636EFA'), (0.2, '#636EFA'),
                         (0.2, '#EF553B'), (0.4, '#EF553B'),
                         (0.4, '#00CC96'), (0.6, '#00CC96'),
                         (0.6, '#AB63FA'), (0.8, '#AB63FA'),
                         (0.8, '#FFA15A'), (1.0, '#FFA15A')]
    sub_region_data = df.groupby("sub-region", dropna=False).count()[["country"]].reset_index()

    # Only return if anything is selected
    if selectedData is not None:
        # Get keys (country name) of all selected points
        indices = []
        for p in selectedData['points']:
            custIndex = p['customdata'][0]
            indices.append(custIndex)

        # Filter for selected points
        df_selected = df[df["country_name"].isin(indices)]
        colorscale = region_colorscale if region_detail == "region" else sub_region_colorscale
        par_coords_column = "region_code" if region_detail == "region" else "sub_region_code"
        print(region_detail)
        print(colorscale)
        fig = go.Figure(data=go.Parcoords(line=dict(color=df_selected[par_coords_column], cauto=False, cmin=0, cmax=1,
                                                    colorscale=colorscale),
                                          dimensions=list([
                                              dict(label='Generosity', range=[df_selected["generosity"].min(),
                                                                              df_selected["generosity"].max()],
                                                   values=df_selected["generosity"]),
                                              dict(label='Freedom', range=[df_selected["freedom"].min(),
                                                                           df_selected["freedom"].max()],
                                                   values=df_selected["freedom"]),
                                              dict(label='GDP', range=[df_selected["gdp_hr"].min(),
                                                                       df_selected["gdp_hr"].max()],
                                                   values=df_selected["gdp_hr"]),
                                              dict(label='Family', range=[df_selected["family"].min(),
                                                                          df_selected["family"].max()],
                                                   values=df_selected["family"]),
                                              dict(label='Health', range=[df_selected["health"].min(),
                                                                          df_selected["health"].max()],
                                                   values=df_selected["health"]),
                                              dict(label='Trust', range=[df_selected["trust"].min(),
                                                                         df_selected["trust"].max()],
                                                   values=df_selected["trust"]),
                                              dict(label='Happiness', range=[df_selected["score"].min(),
                                                                             df_selected["score"].max()],
                                                   values=df_selected["score"]),
                                          ])))

        region_data = df_selected.groupby("region", dropna=False).count()[["country"]].reset_index()
        sub_region_data = df_selected.groupby("sub-region", dropna=False).count()[["country"]].reset_index()
        donut_data = region_data if region_detail == "region" else sub_region_data
        color = region_color if region_detail == "region" else sub_region_color
        fig2 = px.pie(donut_data, values="country", hole=0.4,
                      names=region_detail, color=region_detail,
                      title='Number of Countries per Continent',
                      color_discrete_map=color)

    else:
        par_coords_column = "region_code" if region_detail == "region" else "sub_region_code"
        colorscale = region_colorscale if region_detail == "region" else sub_region_colorscale
        print(par_coords_column)
        print(colorscale)
        fig = go.Figure(data=go.Parcoords(line=dict(color=df[par_coords_column], cauto=False,
                                                    colorscale=colorscale),
                                          dimensions=list([
                                              dict(label='Generosity', range=[df["generosity"].min(),
                                                                              df["generosity"].max()],
                                                   values=df["generosity"]),
                                              dict(label='Freedom', range=[df["freedom"].min(),
                                                                           df["freedom"].max()],
                                                   values=df["freedom"]),
                                              dict(label='GDP', range=[df["gdp_hr"].min(),
                                                                       df["gdp_hr"].max()],
                                                   values=df["gdp_hr"]),
                                              dict(label='Family', range=[df["family"].min(),
                                                                          df["family"].max()],
                                                   values=df["family"]),
                                              dict(label='Health', range=[df["health"].min(),
                                                                          df["health"].max()],
                                                   values=df["health"]),
                                              dict(label='Trust', range=[df["trust"].min(),
                                                                         df["trust"].max()],
                                                   values=df["trust"]),
                                              dict(label='Happiness', range=[df["score"].min(),
                                                                             df["score"].max()],
                                                   values=df["score"]),
                                          ])))

        donut_data = region_data if region_detail == "region" else sub_region_data
        color = region_color if region_detail == "region" else sub_region_color
        fig2 = px.pie(donut_data, values="country", hole=0.4,
                      names=region_detail, color=region_detail,
                      title='Number of Countries per (Sub-)Continent',
                      color_discrete_map=color)

    return fig, fig2


@app.callback([Output('model2', 'figure'),
               Output('model2_results', 'children')], Input('model_set2', 'value'))
def render_model2(model):
    color = '#1F77B4'
    df = c.get_data(model)
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
    return fig, c.generate_result_text(model)


@app.callback([Output('model1', 'figure'), Output('model1_results', 'children')], Input('model_set1', 'value'))
def render_model1(model):
    color = '#FFA15A'
    df = c.get_data(model)
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
    return fig, c.generate_result_text(model)


@app.callback(Output('world-plot', 'figure'),
              [Input('world-year', 'value'), Input("world-value", "value")])
def render_map(year_value, value):
    dff = df[df['year'] == year_value]
    min = df[value].min() - 0.5
    max = df[value].max() + 0.5
    fig = px.choropleth(dff, locations="country", width=1000, height=800,
                        color=value, color_continuous_scale=px.colors.diverging.RdBu[::-1],
                        hover_name="country_name",
                        range_color=(min, max), hover_data=["family",
                                                            'freedom',
                                                            'generosity',
                                                            'score',
                                                            'gdp_hr',
                                                            'trust',
                                                            'dystopia_residual'],
                        locationmode="ISO-3")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(
        coloraxis=dict(colorbar=dict(len=0.7)),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
