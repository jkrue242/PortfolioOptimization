import itertools as it
import pandas as pd 


def get_rolling_correlations(data, LOOKBACK=50):
    # make all combinations of etfs paired together in groups of 2. ex: [('AGG', 'DBC'), ('AGG', 'VTI'), etc]
    etfs_pairs = list(it.combinations(data.columns, 2))
    correlation = pd.DataFrame()

    # build correlation matrix
    for pair in etfs_pairs:
        
        # get dataframe of pair
        pair_df = data[list(pair)]

        # get rolling correlation over 50 day window
        pair_df = pair_df.rolling(LOOKBACK).corr()

        # only keep every other row
        pair_df = pair_df.iloc[0::2,-1] 

        # drop multi-index
        pair_df = pair_df.droplevel(1, axis=0)

        # rename column to be pair
        col_name = str(pair[0])+'-'+str(pair[1])
        correlation[col_name] = pair_df

    rolling_corrs = correlation[LOOKBACK-1:]

    print(rolling_corrs.head())
    return rolling_corrs