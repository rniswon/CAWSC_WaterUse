import os, sys
import numpy as np
import pandas as pd

def filter1 (X, y):

    """

    :param X: original dataframe
    :param y: original tragtes
    :return: X_0, y_0: dropped data
             X_, y_: remaining data
    """

    return X_0, y_0, X_, y_


def drop_abnormal_percapita_wu(X, y, thresh_max, thresh_min):

    drop(X, y)

    return X, y



def get_monthly_frac(df, var_col = '', date_col = 'Date', month_col = None):
    """

    :param df: pandas dataframe with data to calculate average monthly fractions
    :param var_col: the field name of the variable
    :param date_col: date column name
    :param month_col: optional
    :return:
    """

    if not(month_col is None):
        mon_mean = df.groupby(by = [month_col]).mean()
        mon_frac = mon_mean[var_col]/mon_mean[var_col].sum()
        return (mon_mean,mon_frac)
    temp_date = date_col+ "_temp"
    temp_mon = date_col + "_mon"
    df[temp_date] = pd.to_datetime(df[date_col])
    df[temp_mon] = df[temp_date].dt.month

    mon_mean = df.groupby(by=[temp_mon]).mean()
    mon_frac = mon_mean[var_col] / mon_mean[var_col].sum()
    return (mon_mean, mon_frac)


def is_similair(x, y):
    """

    :param x:
    :param y:
    :return:
        return sign_similarity: if close to one means that x and y linearly correlated
        magnitude_similarity

    """
    xis_increasing = np.diff(x)>0
    yis_increasing = np.diff(y)>0

    sign_similarity = np.sum(xis_increasing == yis_increasing)/ len(x)

    Xrel_change =  np.diff(x)/np.sum(x)
    Yrel_change = np.diff(y) / np.sum(y)

    magnitude_similarity = np.mean(Xrel_change/Yrel_change)
    return sign_similarity, magnitude_similarity




def flag_abnormal_wu(X, y, feature_to_compare = 'tmmin'):

    for ws_i in wsas:
        X_for_curr_ent_ws , Y_
        x = get_monthly_frac(X_[tmin])
        y =  get_monthly_frac((y_))

        sim_sig, sim_mag = is_similair(x, y)