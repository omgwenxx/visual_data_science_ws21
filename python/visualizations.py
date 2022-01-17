import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import geopandas

OUTPUTDIR = "img/"
final_df = pd.read_csv("data/final/final_merge_region.csv")

def plot_bp_summary():
    fig, axes = plt.subplots(2, 2, figsize=(30, 20), sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.rcParams["axes.labelsize"] = 20
    fig.suptitle('Happiness Report Summary', fontsize=25)

    sns.boxplot(ax=axes[0, 0], y='family', x='region', data=final_df, width=0.5, palette="colorblind")
    axes[0, 0].set(xlabel='Continent', ylabel='Family')
    axes[0, 0].tick_params(axis="both", labelsize=15)  # x y oder both bei axis
    axes[0, 1].tick_params(axis="both", labelsize=15)
    axes[1, 0].tick_params(axis="both", labelsize=15)
    axes[1, 1].tick_params(axis="both", labelsize=15)

    sns.boxplot(ax=axes[0, 1], y='freedom', x='region', data=final_df, width=0.5, palette="colorblind")
    axes[0, 1].set(xlabel='Continent', ylabel='Freedom')

    sns.boxplot(ax=axes[1, 0], y='generosity', x='region', data=final_df, width=0.5, palette="colorblind")
    axes[1, 0].set(xlabel='Continent', ylabel='Generosity')

    sns.boxplot(ax=axes[1, 1], y='trust', x='region', data=final_df, width=0.5, palette="colorblind")
    axes[1, 1].set(xlabel='Continent', ylabel='Trust')

    fig.savefig(OUTPUTDIR + "boxplot_summary.png", format='png', dpi=100, bbox_inches="tight")

def plot_bp_score():
    fig, ax = plt.subplots()
    bplot = sns.boxplot(y='score', x='region', data=final_df, width=0.5, palette="colorblind")
    bplot.set(xlabel='Continent', ylabel='Happiness Score')
    plt.savefig(OUTPUTDIR + "boxplot_score.png", format='png', dpi=100)

def missing_values():
    plt.figure(figsize=(30, 18))
    sns.color_palette("Greys")
    sns.heatmap(final_df.isna().transpose(),cmap="Greys_r", cbar_kws={'label': 'Missing Data'})
    plt.savefig(OUTPUTDIR + "missing_values.png", dpi=100, orientation="landscape", bbox_inches="tight")

def heatmap():
    # columns missing 90% of data
    missing = final_df.isna().sum(axis=0)
    columns = missing[missing > 1564].keys()  # columns that are missing more than 90% of their values
    filtered_df = final_df.drop(columns.tolist(), axis=1)

    # rows missing 90% of data
    missing = filtered_df.isna().sum(axis=1)
    rows = missing[missing > 514].keys().tolist()
    filtered_df = filtered_df.drop(rows, axis=0)
    data = filtered_df

    plt.figure(figsize=(48, 27))
    palette = sns.diverging_palette(20, 220, n=200)
    palette.reverse()
    data = filtered_df.drop(["country", "year"], axis=1)
    data = data.fillna(0).corr()
    sns.heatmap(data, cmap=palette, yticklabels=False, xticklabels=False, vmin=-1, vmax=1)
    plt.savefig("heatmap.png", dpi=100, orientation="landscape", bbox_inches="tight")
    plt.show()

def bp_ranges(health):
    health_prev = final_df.loc[(final_df["year"] < 2020) & (final_df["year"] >= 2015)].health
    gdp_prev = final_df.loc[(final_df["year"] < 2020) & (final_df["year"] >= 2015)].gdp_hr
    health_after = final_df.loc[final_df["year"] > 2019].health
    gdp_after = final_df.loc[final_df["year"] > 2019].gdp_hr
    health = [health_prev, health_after]
    gdp = [gdp_prev, gdp_after]

    fig, ax = plt.subplots()
    bplot = sns.boxplot(data=health, width=0.5)
    bplot.set(xlabel='Previous years vs. 2020/2021', ylabel='Range')
    plt.savefig(OUTPUTDIR + "boxplot_score_health.png", format='png', dpi=100, bbox_inches="tight")

    fig, ax = plt.subplots()
    bplot = sns.boxplot(data=gdp, width=0.5)
    bplot.set(xlabel='Previous years vs. 2020/2021', ylabel='Range')
    plt.savefig(OUTPUTDIR + "boxplot_score_gdp.png", format='png', dpi=100, bbox_inches="tight")

def plot_continent_values():
    # countries per continent
    df_unique_countries = final_df.drop_duplicates(subset="country")[["region", "country"]]
    df_unique_countries.groupby("region").count()

    final_df.groupby("region")[["country", "score"]].mean()
    final_df.groupby("region")[["country", "score"]].max()
    final_df.groupby("region")[["country", "score"]].min()

def plot_lineplots():
    # plot continent over time
    # plot sub-regions over time
    value_per_continent = final_df.groupby(["region", "year"]).mean()
    value_per_subcontinent = final_df.groupby(["sub-region", "year"]).mean()
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)
    sns.lineplot(x='year', y='score', hue="region", data=value_per_continent, ax=ax1)
    sns.lineplot(x='year', y='score', hue="sub-region", data=value_per_subcontinent, ax=ax2)

    ax1.title.set_text("Continents")
    ax1.set_ylabel("Happiness Score")
    ax1.set_xlabel("Year")
    ax1.tick_params(axis="y", labelsize=15)
    ax1.tick_params(axis="x", labelsize=13)
    ax2.tick_params(axis="x", labelsize=13)
    ax2.title.set_text('Sub-Regions')
    ax2.set_xlabel("Year")

    ax1.legend(bbox_to_anchor=(1.25, 1), loc='upper right', borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.savefig(OUTPUTDIR + '/lineplots.png', bbox_inches="tight")
    plt.show()

def plot_world(plot_2015 = False):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # rename the columns so that we can merge with our data
    world.columns = ['pop_est', 'continent', 'name', 'country', 'gdp_md_est', 'geometry']
    # then merge with our data
    df_2021 = final_df.loc[final_df.year == 2015][["country", "score"]]
    df_2015 = final_df.loc[final_df.year == 2021][["country", "score"]]
    merge = pd.merge(world, df_2021, on='country', how="left")

    print("Plotting world map")
    filename = 'world_2021.png'
    title = 'Happiness Score 2021'
    vmin = 2.5
    vmax = 8
    if plot_2015:
        merge = pd.merge(world, df_2015, on='country', how="left")
        filename = 'world_2015.png'
        title = 'Happiness Score 2015'

    # plot confirmed cases world map
    ax = merge.plot(column='score',
                    figsize=(25, 20),
                    legend=True, cmap='PiYG',
                    vmin=vmin, vmax=vmax,
                    missing_kwds={'color': 'lightgrey'})
    fig = ax.figure
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=20)

    plt.title(title, fontsize=25)
    plt.savefig(OUTPUTDIR + filename, bbox_inches="tight")

def plot_world_gdp():
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # rename the columns so that we can merge with our data
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    # calculate GDP per capita by dividing GDP by population size
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est

    plot_2015 = False
    filename = 'world_2021.png'
    title = 'GDP Per Capita'
    plt.subplots_adjust(hspace=0.1)

    # plot confirmed cases world map
    ax = world.plot(column='gdp_per_cap',
                    figsize=(25, 20),
                    legend=True, cmap='OrRd',
                    missing_kwds={'color': 'lightgrey'},
                    legend_kwds={'pad': 0.01, 'orientation': "horizontal"})
    fig = ax.figure
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=20)

    plt.title(title, fontsize=25)
    plt.savefig(OUTPUTDIR + "gdp", bbox_inches="tight")

def corr_heatmap():
    # columns missing 90% of data
    missing = final_df.isna().sum(axis=0)
    columns = missing[missing > 1564].keys()  # columns that are missing more than 90% of their values
    filtered_df = final_df.drop(columns.tolist(), axis=1)

    # rows missing 90% of data
    missing = filtered_df.isna().sum(axis=1)
    rows = missing[missing > 514].keys().tolist()
    filtered_df = filtered_df.drop(rows, axis=0)
    data = filtered_df

def plot_scatterplot():
    pd.set_option("max_rows", 10)
    wdi_df = pd.read_csv("data/final/wdi_final.csv")
    wdi_columns = wdi_df.columns

    edu_df = pd.read_csv("data/final/edu_stats.csv")
    edu_columns = edu_df.columns

    # Select rows which do not have NaN value in column 'score'
    selected_rows = final_df[~final_df['score'].isnull()]
    selected_rows = selected_rows.groupby("country").mean()

    # only keep columns that have at least 50 values
    final_corr = selected_rows.dropna(axis=1, how='any', thresh=50, subset=None, inplace=False)
    wdi_columns = list(set(final_corr.columns).intersection(wdi_columns))
    wdi_columns.append("score")
    wdi_set = final_corr[list(set(final_corr.columns).intersection(wdi_columns))]
    wdi_corr = wdi_set.corr()
    wdi_corr[wdi_corr["score"].abs() >= 0.7]["score"]

    columns = list(set(final_corr.columns).intersection(edu_columns))
    columns.append("score")
    edu_set = selected_rows[columns]
    edu_set.dropna(axis=1, how='any', thresh=140, subset=None, inplace=False)
    edu_corr = edu_set.corr()
    edu_values = edu_corr[edu_corr["score"].abs() >= 0.75]["score"]