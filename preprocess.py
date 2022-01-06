import pandas as pd
import json
import numpy as np

def get_countries_per_continent(final_df):
    df_unique_countries = final_df.drop_duplicates(subset="country")[["region", "country"]]
    df_unique_countries.groupby("region").count()

def merge_region():
    final_df = pd.read_csv("data/final/final_merge.csv")
    iso_region = pd.read_csv("data/iso-codes-with-region.csv")

    # applying merge
    iso_region = iso_region.rename(columns={"alpha-3": "country"})
    # left to keep rows from dataset merge
    final_region = final_df.merge(iso_region[["country", "region", "sub-region"]], on='country', how="left")
    final_region.to_csv('data/final/final_merge_region.csv',index=False)

def impute_missing_values_hr(temp):
    """
    This menthod imputes missing data with the data of the next year
    """
    temp.reset_index(inplace=True)  # reset multilevel index
    temp.sort_values(["country", "year"], ignore_index=True,
                     inplace=True)  # sort by country and year for filling in next year
    temp.fillna(method="ffill", inplace=True)  # fill in value form next year 2019
    temp.set_index(['country', 'year'], inplace=True)  # set multilevel index again

def set_index_to_iso(dataset):
    """
    This method sets all names to ISO code standard e.g. Austria to AUT.
    """
    iso_codes = pd.read_csv("data/iso-country-codes.csv")
    iso_codes = dict(zip(iso_codes['English short name lower case'].str.lower(), iso_codes['Alpha-3 code']))
    iso_codes["united states"] = "USA"
    iso_codes["south korea"] = "KOR"
    dataset['country'] = dataset.index.get_level_values(0)
    dataset['country'] = dataset['country'].str.lower().map(iso_codes)
    countries = dataset['country'].notnull()

    countries_df = dataset[countries]
    countries_df['year'] = countries_df.index.get_level_values(1)
    countries_df = countries_df.set_index(['country', "year"])
    return countries_df

def preprocess_hr(save:bool=False):
    """
    This method includes all the necessary data preprocessing for the World Happiness Report data (2015-2021)
    """
    with open('helper_data/happinessreport_mapping.json', 'r') as openfile:
        hr_mapping = json.load(openfile)

    # reverse mapping key-> values becomes  values[0]-> key, values[1]->key, ...
    hr_mapping_reversed = {}
    for key in hr_mapping.keys():
        values = hr_mapping[key]
        for value in values:
            hr_mapping_reversed[value] = key

    hr_mapping_elements = []
    for value in hr_mapping.values():
        hr_mapping_elements += value

    hr_frames = []
    for year in range(2015, 2022):
        temp = pd.read_csv("data/world_happiness_report/%d.csv" % year)

        # avoid mapping issues
        if year == 2021:
            temp.drop(columns=["Explained by: Perceptions of corruption", "Explained by: Healthy life expectancy",
                               "Explained by: Generosity",
                               "Explained by: Freedom to make life choices", "Explained by: Social support",
                               "Explained by: Log GDP per capita"], inplace=True)

        relevant_cols = sorted(list(set(hr_mapping_elements).intersection(set(temp.columns))))
        temp = temp.loc[:, relevant_cols]
        temp.rename(columns={col: hr_mapping_reversed[col] for col in relevant_cols}, inplace=True)
        # sort columns
        temp = temp.loc[:, sorted([hr_mapping_reversed[col] for col in relevant_cols])]
        temp["year"] = year
        temp.set_index(['country', 'year'], inplace=True)
        hr_frames.append(temp)
        print(str(year) + " missing columns")
        print(set(hr_mapping.keys()).difference(set(temp.columns).union(["region"])))

    hr_df = pd.concat(hr_frames)
    impute_missing_values_hr(hr_df)

    hr_df = set_index_to_iso(hr_df)

    if save:
        hr_df.to_csv('data/final/hr_final.csv')

    return hr_df

def preprocess_wdi(save:bool=False):
    """
    This method includes all the necessary data preprocessing for the World Development Indicator data (2015-2020)
    """
    wdi_df = pd.read_csv("data/wdi/wdi.csv")
    wdi_df.drop(wdi_df.tail(5).index, inplace=True)
    years = list(range(2015, 2021))
    cols = list(wdi_df.columns[0:4]) + years
    wdi_df.columns = cols
    wdi_df.drop(['Country Name', 'Series Code'], axis=1, inplace=True)
    wdi_df = wdi_df.rename(columns={'Country Code': "country"})
    wdi_df = wdi_df.set_index(['country', 'Series Name'])
    wdi_df = wdi_df.stack()
    cols = ['country', 'Series Name', 'year']
    wdi_df.index.names = cols
    wdi_df = wdi_df.unstack('Series Name')
    if save:
        wdi_df.to_csv('data/final/wdi_final.csv')

    return wdi_df

def preprocess_edu(save:bool=False):
    """
    This method includes all the necessary data preprocessing for the Education statistics data (2015-2020)
    """
    edu_df = pd.read_csv("data/edu_stats/EdStatsData.csv")
    year_drop = list(map(str, range(1970, 2015, 1)))
    year_drop2 = list(map(str, range(2025, 2101, 5)))
    years = list(range(2015, 2018)).append(2020)
    edu_df.drop(year_drop, axis=1, inplace=True)
    edu_df.drop(year_drop2, axis=1, inplace=True)
    edu_df.drop(['Country Name', 'Indicator Code', 'Unnamed: 69'], axis=1, inplace=True)
    edu_df = edu_df.rename(columns={'Country Code': "country"})
    edu_df = edu_df.set_index(['country', 'Indicator Name'])
    edu_df = edu_df.stack()
    cols = ['country', 'Indicator Name', 'year']
    edu_df.index.names = cols
    edu_df = edu_df.unstack('Indicator Name')

    if save:
        edu_df.to_csv('data/final/edu_stats.csv')

    return edu_df

def merge(save:bool = False):
    """
    Merge datasets to one final dataset.
    """
    edu_df = pd.read_csv("data/final/edu_stats.csv")
    wdi_df = pd.read_csv("data/final/wdi_final.csv")
    hr_df = pd.read_csv("data/final/hr_final.csv")
    edu_df.set_index(['country', 'year'], inplace=True)
    wdi_df.set_index(['country', 'year'], inplace=True)
    hr_df.set_index(['country', 'year'], inplace=True)
    hr_wdi_df = pd.merge(hr_df, wdi_df, on=["country", "year"], how="outer", suffixes=("left", ""))
    final_merge = pd.merge(hr_wdi_df, edu_df, on=["country", "year"], how="outer",suffixes=("left", ""))

    # fix mixed columns that have string and float value, convert strings of ".." to NaN values
    corr_columns = final_merge.select_dtypes(['object']).columns
    final_merge[corr_columns] = pd.to_numeric(corr_columns, errors='coerce')


    if save:
        final_merge.to_csv('../data/final/final_merge.csv')


def create_files(new: bool = False):
    """
    Creates files of the different data sources and preprocessed them in order to finally merge them.
    After this method is called, 5 files will be created: happiness report (merged over all years) as hr_report.csv,
    educational statistics as edu_stats.csv, world development indicators as wdi_final.csv and the merged files with region
    information (final_merge_region.csv) and without region information (final_merge.csv).
    :param new: bool
        Set to True if files should be saved.
    """
    if new:
        preprocess_hr(True)
        preprocess_edu(True)
        preprocess_wdi(True)
        merge(True)
        merge_region()
    else:
        preprocess_hr()
        preprocess_edu()
        preprocess_wdi()
        merge()
        merge_region()

