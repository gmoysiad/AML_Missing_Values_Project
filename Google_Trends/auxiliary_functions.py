"""
Python file that contains functions that help other major classes
"""

import itertools as it
import time
from datetime import datetime, timedelta

import pandas as pd
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq


def week_frequency(days: int) -> str:
    """Based on the number of days passed it returns the corresponding frequency of weeks"""
    if days <= 7:
        return '_weekly'
    elif 14 >= days >= 7:
        return '_biweekly'
    elif 21 >= days >= 14:
        return '_triweekly'
    else:
        return '_monthly'


def transform_dataframe(df: pd.DataFrame, frequency) -> pd.DataFrame:
    """Given a dataframe and a frequency that the data was captured returns a transposed dataframe with additional
    indexes indicating also the frequency that the data was captured."""
    frequency = week_frequency(frequency)
    frequency = [[frequency] * len(df)][0]
    end_indexes = list(map(aux_timestamps, df.index))
    tuple_indexes = list(it.zip_longest(frequency, df.index, end_indexes))
    indexes = pd.MultiIndex.from_tuples(tuple_indexes, names=['Frequency', 'Start', 'End'])
    df = df.set_index(indexes)
    return df.transpose()


def aux_timestamps(start_index: datetime) -> datetime:
    return start_index + timedelta(seconds=59, minutes=59, hours=23)


def fetch_queries(query: str, timeframe: str, location='GR'):
    """Returns from Google Trends a dictionary of dataframes containing the related queries of a query during a
    specific timeframe based on a geolocation.

    Parameters
    ----------
    query:
        string - query-search_term that will base the data collection
    timeframe:
        string - we pass a timeframe that has been operated on with datetime, and then we convert it to string
    location:
        string - the location we want to base our search, e.g. a country or a worldwide base search

    Returns
    -------
    related_queries:
        DataFrame - contains the related_queries of a query for the specific timeframe
    """
    pytrends = TrendReq()
    try:
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)
    except ResponseError:
        print('Too many requests to Google Trends. Google Trends disconnected.')
    except ConnectionError:
        print('Connection timed out, please wait...')
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)

    return pytrends.related_queries()


def fetch_dataframes(query: str, timeframe: str, location: str = 'GR'):
    """Returns from Google Trends a dataframe containing the interest_over_time of a query during a specific
    timeframe based on a geolocation. If, during that timeframe, it returns an empty dataframe, because there was no
    data to be found, we create our own dataframe containing zeros for that timeframe.

    Parameters
    ----------
    query:
        string - query-search_term that will base the data collection
    timeframe:
        string - we pass a timeframe that has been operated on with datetime, and then we convert it to string
    location:
        string - the location we want to base our search, e.g. a country or a worldwide base search

    Returns
    -------
    int_over_time:
        DataFrame - contains the interest_over_time (popularity) of a query for the specific timeframe
    """
    pytrends = TrendReq()
    # print(query, timeframe, location)
    try:
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)
        time.sleep(5)
    except ResponseError:
        print('Too many requests to Google Trends. Google Trends disconnected.')
    except ConnectionError:
        print('Connection timed out, please wait...')
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)

    # if a week has 0 interest_over_time pytrends will return an empty dataframe, and it's raising KeyError because
    # it's an empty dataframe there is no column=isPartial for it to drop
    try:
        int_over_time = pytrends.interest_over_time()
        int_over_time.drop('isPartial', axis=1, inplace=True)
    except KeyError:
        # create an index based on the timeframe that we have passed above
        str_idx = datetime.strptime(timeframe.split(' ')[0], '%Y-%m-%d')
        end_idx = datetime.strptime(timeframe.split(' ')[1], '%Y-%m-%d')
        index = pd.date_range(str_idx, end_idx, freq='D')

        # create and return a dataframe with 0s based on the index that we created above
        int_over_time = pd.DataFrame(index=index, columns=[query]).fillna(0)
    return int_over_time
