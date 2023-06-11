"""
Converts raw Google Trends data to rescaled and filtered Google Trends approximating a monthly trend for a given query
based on a geo location that the query was searched.
"""

from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY, WEEKLY

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import models
import auxiliary_functions as af
from DTW.DTW import DynamicTimeWarping


class Query:
    """
    Class Query holds in an object-type a query, the town that it's based on, its rescaled values based on a combination
    of regression models we passed as parameters and its training period, its aggregated data from the combination of
    regression models used and average data based on the weights of each regression model
    """
    _default = 7  # number of days that we will use for the timedelta operations later on
    _steps = [1, 2, 3, 4, 21, 26]  # number of steps that will skip in the date loop
    _frequencies = [i * 7 for i in _steps]  # number of days that correspond to 1, 2, 3, 4, etc. weeks

    def __init__(self,
                 query: str = None,
                 start: datetime = datetime(2015, 1, 1),  # random starting date
                 end: datetime = datetime.today(),
                 days: int = _default,
                 origin: str = 'GR',
                 **kwargs):
        """
        Parameters
        ----------
        query:
            str
        start:
            date
        end:
            date
        days:
            int
        origin:
            str
        """
        # print(query, start, end, days, origin)
        self.__query = query
        self.__language_code = query[1]
        self.__destination = self.__query.split()[1]
        self.__start = start
        self.__end = end
        self.__origin = origin
        self.__model = models.LSTM(**kwargs)
        t_f = self.__start.strftime('%Y-%m-%d') + ' ' + self.__end.strftime('%Y-%m-%d')
        self.__timeline = af.fetch_dataframes(self.__query, t_f, location=self.__origin)
        # self._dtw = ErrorWrapper.error(self._database, self._timeline, case='dtw')

    def rescale(self, data: pd.DataFrame,
                start_date: datetime,
                end_date: datetime,
                query: str,
                days: int = _default) -> pd.DataFrame:
        """Creates uniformly scaled Google Trends data.

        A TensorFlow Neural Network model rescales the raw Google Trends data that are fetched and stored in such way
        directly from Google. Each frequency that the data are stored and fetched have different scales based on the
        number of searches that a query had for that period of time.

        In order to eliminate this "Google scale" we train Neural Network models that do this exact job. In order to be
        as accurate as it can be the raw Google Trends data are fetched and stored in many different frequencies in
        order to ensure the different multiple scales that Google has for a certain date in different time frames.

        Lastly, the models reset after a certain period of time in order to ensure and avoid an "overfitting trend",
        along with that each different period of time has different a different trend, e.g. a summer specified search
        has a greater trend during the spring period since people are search for something to do in the summer that is
        to come.

        Parameters
        ----------
        data:
            DataFrame - contains the raw Google Trends data
        start_date:
            date - date indicating the start of the data
        end_date:
            date - date indicating the end of the data
        query:
            str - query that was used to collect trends data
        days:
            int - number indicating the way that the data is fetched (7=weekly, 14=biweekly, etc.)

        Returns
        -------
        database:
            pd.DataFrame - a pandas DataFrame that contains the rescaled Google Trends data
        """
        interval = calculate_interval(self._default)

        database = pd.DataFrame()

        for month_start in rrule(freq=MONTHLY, dtstart=start_date, interval=interval, until=end_date):
            month_end = calculate_month_end(month_start, end_date, interval)

            first_week_end_dt = month_start + timedelta(days=days - 1)  # end of first week
            first_week_tf = month_start.strftime('%Y-%m-%d') + ' ' + first_week_end_dt.strftime('%Y-%m-%d')
            first_week = af.fetch_dataframes(query=query, timeframe=first_week_tf)  # data.loc[month_start:first_week_end_dt]

            database = database.append(first_week)

            second_week_start_dt = first_week_end_dt + timedelta(days=1)
            second_week_end_dt = calculate_auxiliary_date(second_week_start_dt, days, end_date)
            second_week_tf = second_week_start_dt.strftime('%Y-%m-%d') + ' ' + second_week_end_dt.strftime('%Y-%m-%d')
            second_week = af.fetch_dataframes(query=query, timeframe=second_week_tf)  # data.loc[second_week_start_dt:second_week_end_dt]
            t_f = month_start.strftime('%Y-%m-%d') + ' ' + second_week_end_dt.strftime('%Y-%m-%d')

            auxiliary_table = af.fetch_dataframes(self.__query, t_f, location=self.__origin)

            self.__model.sampling(second_week, auxiliary_table, first_week)
            self.__model.build()
            self.__model.train()

            scaled = self.__model(second_week)

            database = database.append(scaled)

            aux_start = second_week_end_dt + timedelta(days=1)

            if month_end > end_date:
                month_end = end_date
            for current_date in rrule(freq=WEEKLY, dtstart=aux_start, until=month_end):
                d1 = current_date
                d2 = calculate_d2(d1, month_end)
                tf = d1.strftime('%Y-%m-%d') + ' ' + d2.strftime('%Y-%m-%d')
                next_week = af.fetch_dataframes(query=query, timeframe=tf)  # data.loc[d1:d2]
                # print(tf)
                # print(next_week)
                scaled = self.__model(next_week)
                database = database.append(scaled)

        return database

    '''def update(self,
               data: pd.DataFrame,
               end: datetime = datetime.today(),
               days: int = _default) -> pd.DataFrame:
        """Updates until an end date the uniformly scaled Google Trends data.

        Function that appends into the "data" DataFrame the newly fetched Google Trends data, rescales them to a uniform
        scale and stores both the scaled and the raw data.

        Parameters
        ----------
        data:
            DataFrame - outdated data that is already stored in the database and needs to be updated with data until
            the today's or a different date.
        end:
            date - a date that indicates the end date of the update for the data, default is set to today's date,
            though a different day is also possible.
        days:
            int - number of days indicating the frequency in which the incoming data will be, it may vary as well

        Returns
        -------
        data: a pandas DataFrame that contains the updated uniformly scaled data.
        """
        # print('I am inside the update function')
        # raw_data = self._raw_data
        start = data.index[-1]
        interval = calculate_interval(days)
        self.__end = end

        for month_start in rrule(freq=MONTHLY, dtstart=start, interval=interval, until=end):
            month_end = calculate_month_end(month_start, end, interval)

            first_week_start_dt = month_start - timedelta(days=days - 1)  # end of first week
            first_week = data.loc[first_week_start_dt:month_start]

            second_week_start_dt = (month_start + timedelta(days=1)).strftime('%Y-%m-%d')
            second_week_end_dt = (second_week_start_dt + timedelta(days=days - 1)).strftime('%Y-%m-%d')
            second_week = af.fetch_dataframes(self.__query, second_week_start_dt + ' ' + second_week_end_dt,
                                              self.__origin)

            auxiliary_table = af.fetch_dataframes(self.__query, first_week_start_dt.strftime('%Y-%m-%d') + ' ' +
                                                  second_week_end_dt, location=self.__origin)
            lstm = models.LSTM()
            lstm.sampling(second_week, auxiliary_table, first_week)
            lstm.build()
            lstm.train()
            scaled = lstm(second_week)
            data = data.append(scaled)

            aux_start = second_week_end_dt + timedelta(days=1)
            for current_date in rrule(freq=WEEKLY, dtstart=aux_start, until=month_end):
                d1 = current_date
                d2 = calculate_d2(d1, month_end)

                next_week = data.loc[d1:d2]
                scaled = lstm(next_week)
                data = data.append(scaled)

        return data  # , raw_data'''

    @property
    def query(self):
        return self.__query

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def timeline(self):
        return self.__timeline


class ErrorWrapper:
    """
    Class that calls upon the error functions to calculate between 2 time series, in our case the isolated/raw data
    with some scaled data
    """

    def r2(*time_series):
        """r^2 error calculator"""
        return r2_score(*time_series)

    def dtw(*time_series, bag_size=60):
        """Dynamic Time Warping error calculator"""
        DTW = DynamicTimeWarping()
        DTW.get(time_series[0], time_series[1], bucket_size=bag_size)
        return DTW.D

    def pearson(*time_series):
        """Pearson' correlation coefficient calculator"""
        return pearsonr(*time_series)

    error_functions = {'r2': r2,
                       'dtw': dtw,
                       'pearson': pearson}

    @staticmethod
    def error(time_series1, time_series2, case='dtw'):
        """Returns a specified error for 2 time series

        Parameters
        ----------
        time_series1:
            nparray - contains the scaled or raw data
        time_series2:
            nparray - contains the scaled or raw data
        case:
            string - contains the method that will calculate the error or correlation between the pair of data
        """
        func = ErrorWrapper.error_functions.get(case)
        ts1 = time_series1.values.reshape((1, -1))
        ts2 = time_series2.values.reshape((1, -1))
        return list(map(func, ts1, ts2))[0]


def calculate_interval(days: int) -> int:
    """If the number of days is greater than 14 (2 weeks period) then it is safer to use a bi/tri-monthly period of
    train"""
    if days == 14:
        return 2
    elif days in (21, 28):
        return 3
    else:
        return 1


def calculate_month_end(month_start: datetime, end_date: datetime, interval: int) -> datetime:
    """Calculates whether the end of the month date is either the end date or not"""
    if (month_start - end_date).days in (-29, 30):
        return end_date
    else:
        return month_start + relativedelta(months=interval) - timedelta(days=1)


def calculate_auxiliary_date(d1: datetime, days: int, month_end: datetime) -> datetime:
    """Calculates the end date of the incoming week so that it does not overstep with the end of the month date"""
    d2 = d1 + timedelta(days=days - 1)
    if d2 > month_end:
        d2 = month_end
    return d2


def calculate_d2(d1: datetime, end_date: datetime) -> datetime:
    """Calculates the end date of the second week that we are going to append into our database so that it does not
    overstep with the ending date"""
    d2 = d1 + timedelta(days=6)
    if d2 > end_date:
        d2 = end_date
    return d2



