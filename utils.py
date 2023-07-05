import pandas as pd

def make_columns_numerical(d:pd.DataFrame, in_place =False):
    obj_cols = d.select_dtypes(include='object').columns
    d_copy = d
    if not in_place:
        d_copy = d.copy()
    d_copy[obj_cols] = d_copy[obj_cols].apply(lambda col:pd.Categorical(col).codes)
    return d_copy

def make_summary_of_df(d:pd.DataFrame):
    a = pd.DataFrame(data=[d.dtypes, d.count(), d.nunique(), d.min(), d.max(), d.isna().sum()], columns=d.columns)
    a.index = ['dtype','count', 'nunique', 'min', 'max', 'missing values']
    return a


def make_percentage(d:pd.DataFrame, col_name, coly='Attrition', bins=10,bin_over_20 =True):
    dat = d[[col_name, coly]].copy()
    if bin_over_20 and dat[col_name].nunique() > 20:
        dat[col_name] = pd.cut(dat[col_name], bins=bins, retbins=True)[0]
    a = dat.groupby(by=[col_name,coly], as_index=False).size()
    totals = dat.groupby(by=[col_name], as_index=False).size()

    a['%total'] = a['size'] / dat[col_name].count()
    a2 = a.merge(totals, on=col_name)
    a2['%part'] = a2.size_x / a2.size_y
    return a2[[col_name, coly, '%part', '%total']]
