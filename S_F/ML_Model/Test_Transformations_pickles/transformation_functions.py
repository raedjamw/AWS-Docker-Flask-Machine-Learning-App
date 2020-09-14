import pandas as pd

def dummies_funct(df,prefix):
    """
    Binarize the nominal categorical data and include a nummy column
    """
    if df.dtype != 'object':
        raise TypeError('dtype must be of type "object"')

    dummies_df = pd.get_dummies(df, drop_first=True, prefix=prefix, prefix_sep='_', dummy_na=True)
    return dummies_df


def imputer_func(df, strategy, removed):
    from sklearn.impute import SimpleImputer
    import numpy as np
    """
    Test imputer function converts column nans to means

    """
    if df.dtypes.value_counts().dtypes != 'int64':
        raise TypeError('dtype must be be numeric')

    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputed_df = pd.DataFrame(imputer.fit_transform(df.drop(columns=removed)), columns=df.drop(columns=removed).columns)
    return imputed_df