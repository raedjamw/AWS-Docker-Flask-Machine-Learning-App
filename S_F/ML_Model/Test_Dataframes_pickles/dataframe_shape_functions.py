import pandas as pd


# load the datasets
def df_from_csv(filename):
    """
    Return a df read from a csv

    """
    return pd.read_csv(filename)


def series_df_concat(df1,df2,drop_Val):
  """
  Concat dataframes along columns

  """
  if df2.name in df1.columns:
    raise ValueError('column already exists')

  concat_df = pd.concat([df1, df2], axis=1, sort=False).reset_index(drop=drop_Val)

  return concat_df


def df_df_concat(df1,df2):
  """
  Concat dataframes along columns

  """
  if set(df2.columns).issubset(df1.columns):
    raise ValueError('columns already exists')

  concat_df = pd.concat([df1, df2], axis=1, sort=False)

  return concat_df


def df_df_row_concat(df1,df2):
  """
  Concat dataframes along columns

  """
  if pd.merge(df2,df1).equals(df2) == True:
    raise ValueError('columns already exists')

  concat_df = pd.concat([df1, df2], axis=0)

  return concat_df