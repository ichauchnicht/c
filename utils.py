import pandas as pd


def make_columns_numerical(d: pd.DataFrame, in_place=False):
    '''
    transforms columns of dtype object into number using pandas categorical datatype
    todo use label encoder
    :param d: pandas dataframe
    :param in_place: if True return the same object
    :return: a copy or the same transformed object
    '''
    obj_cols = d.select_dtypes(include='object').columns
    d_copy = d
    if not in_place:
        d_copy = d.copy()
    d_copy[obj_cols] = d_copy[obj_cols].apply(lambda col: pd.Categorical(col).codes)
    return d_copy


def make_summary_of_df(d: pd.DataFrame):
    '''
    for the given dataframe, make a summary containing all columns dtypes, count, number of unique values, min and max value, number of missing values
    :d pandas dataframe
    :returns a dataframe containing summary values for the given dataframe.
    '''
    a = pd.DataFrame(data=[d.dtypes, d.count(), d.nunique(), d.min(), d.max(), d.isna().sum()], columns=d.columns)
    a.index = ['dtype', 'count', 'nunique', 'min', 'max', 'missing values']
    return a


def make_percentage(d: pd.DataFrame, col_name, coly='Attrition', bins=10, bin_over_20=True):
    '''
    calculaled totals and percentages for aggregation for the given columns col_name and coly
    :param d: the source data frame
    :param col_name: the column to be aggregated with
    :param coly: the 2nd column to be aggregated with, default='Attrition'
    :param bins: number of bins, default is 10
    :param bin_over_20: if True and the column col_name has more than 20 values, bin it. default:True
    :return: a daframe containing totals and percentages for aggregation for the given columns
    '''
    dat = d[[col_name, coly]].copy()
    if bin_over_20 and dat[col_name].nunique() > 20:
        dat[col_name] = pd.cut(dat[col_name], bins=bins, retbins=True)[0]
    a = dat.groupby(by=[col_name, coly], as_index=False).size()
    totals = dat.groupby(by=[col_name], as_index=False).size()

    a['%total'] = a['size'] / dat[col_name].count()
    a2 = a.merge(totals, on=col_name)
    a2['%part'] = a2.size_x / a2.size_y
    return a2[[col_name, coly, '%part', '%total']]
