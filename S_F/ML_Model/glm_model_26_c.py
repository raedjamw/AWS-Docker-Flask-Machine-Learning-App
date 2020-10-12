# std packages
import numpy as np
import pandas as pd
import pickle
from collections import Counter

# ML packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import statsmodels.api as sm

# User Defined Modules
import Test_Dataframes_pickles.dataframe_shape_functions as dfsf
import Test_Transformations_pickles.transformation_functions as tff

# function to get csv as dataframe
raw_train = dfsf.df_from_csv(r'exercise_26_train.csv')

# Describing the target variable
Counter(raw_train.y)
#
# Overview of data types
print("object dtype:", raw_train.columns[raw_train.dtypes == 'object'].tolist())
print("int64 dtype:", raw_train.columns[raw_train.dtypes == 'int'].tolist())
print("The rest of the columns have float64 dtypes.")

# Investigate Object Columns
def investigate_object(df):
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = df.columns[df.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(df[col_obj[i]].unique()) > 13:
            print(col_obj[i]+":", "Unique Values:", np.append(df[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i]+":", "Unique Values:", df[col_obj[i]].unique())

    del col_obj


"""
Feature Engineering
"""

# copy train set
train_val = raw_train.copy(deep=True)

#1 Formatting to numeric
train_val['x12'] = train_val['x12'].str.replace('$','')
train_val['x12'] = train_val['x12'].str.replace(',','')
train_val['x12'] = train_val['x12'].str.replace(')','')
train_val['x12'] = train_val['x12'].str.replace('(','-')
train_val['x12'] = train_val['x12'].astype(float)
train_val['x63'] = train_val['x63'].str.replace('%','')
train_val['x63'] = train_val['x63'].astype(float)

# train test split the Training data
x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)


# 2. smashing sets back together
train = dfsf.series_df_concat(x_train, y_train,True)
val = dfsf.series_df_concat(x_val, y_val, True)
test = dfsf.series_df_concat(x_test, y_test, True)

# 3. With mean imputation from Train set
train_imputed = tff.imputer_func(train, 'mean', ['y', 'x5', 'x31', 'x81','x82'])
print(train_imputed.shape)
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)
print(train_imputed_std.shape)

# Saving the train columns to disk as pkl.
train_columns = train_imputed.columns
train_col_fl = 'prediction_pickles/train_columns.pkl'
pickle.dump(train_columns, open(train_col_fl, 'wb'))
print("[INFO]: Finished saving train_columns...")

# Save std scaler function to disk as pkl.
std_scaler_fl = 'prediction_pickles/fitted_std_scaler.pkl'
pickle.dump(std_scaler, open(std_scaler_fl, 'wb'))
print("[INFO]: Finished saving fitted_std_scaler...")

"""
convert nominal categorical variables to numeric via 1 hot encoding

"""

# apply dummies to x5
dumb5 = tff.dummies_funct(train['x5'],'x5')
# concat to rest of the dataframe
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb5)

# apply dummies to x31
dumb31 = tff.dummies_funct(train['x31'],'x31')
# print(dumb31.columns)
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb31)

dumb81 = tff.dummies_funct(train['x81'],'x81')
# print(dumb81.columns)
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb81)

dumb82 = tff.dummies_funct(train['x82'],'x82')
# print(dumb82.columns)
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb82)

train_imputed_std = dfsf.series_df_concat(train_imputed_std, train['y'],True)

"""
Initial Feature Selection
"""
exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')
print(var_reduced)
variables = var_reduced['name'].to_list()


# Saving the variables to disk
variables_fl = 'prediction_pickles/variables.pkl'
pickle.dump(variables, open(variables_fl, 'wb'))
print("[INFO]: Finished saving variables...")
print(variables)
print('length of vars', len(variables))
# Preliminary Model

"""
Starting with the train set. The L1 process creates biased parameter estimates.
As a result, we will build a final model without biased estimators.
"""
logit = sm.Logit(train_imputed_std['y'], train_imputed_std[variables])
# fit the model
result = logit.fit()
result.summary()




# Prepping the validation set

val_imputed = tff.imputer_func(val, 'mean', ['y', 'x5', 'x31',  'x81' ,'x82'])

val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=train_imputed.columns)

# convert nominal categorical variables to numeric via 1 hot encoding
dumb5 = tff.dummies_funct(val['x5'],'x5')
# print(dumb5.columns)
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb5)

dumb31 = tff.dummies_funct(val['x31'],'x31')
# print(dumb31.columns)
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb31)

dumb81 = tff.dummies_funct(val['x81'],'x81')
# print(dumb81.columns)
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb81)

dumb82 = tff.dummies_funct(val['x82'],'x82')
# print(dumb82.columns)
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb82)
val_imputed_std  = dfsf.series_df_concat(val_imputed_std, val['y'],True)


# Prepping the test set

test_imputed = tff.imputer_func(test,'mean',['y', 'x5', 'x31',  'x81' ,'x82'])

test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=train_imputed.columns)

# # convert nominal categorical variables to numeric via 1 hot encoding
dumb5 = tff.dummies_funct(test['x5'],'x5')
# print(dumb5.columns)
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb5)

dumb31 = tff.dummies_funct(test['x31'],'x31')
# print(dumb31.columns)
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb31)

dumb81 = tff.dummies_funct(test['x81'],'x81')
# print(dumb81.columns)
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb81)

dumb82 = tff.dummies_funct(test['x82'],'x82')
# print(dumb82.columns)
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb82)
test_imputed_std = dfsf.series_df_concat(test_imputed_std, test['y'],True)

# Models Results with the train set
Outcomes_train = pd.DataFrame(result.predict(train_imputed_std[variables])).rename(columns={0:'probs'})
# print(Outcomes_train)
# print('##########################')
Outcomes_train['y'] = train_imputed_std['y']
#print(Outcomes_train['y'])
#print('The C-Statistics is ',roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']))

# Models Results with the val set
Outcomes_val = pd.DataFrame(result.predict(val_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_val['y'] = val_imputed_std['y']
# print('The C-Statistics is ',roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']))


# Models Results with the test set
Outcomes_test = pd.DataFrame(result.predict(test_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_test['y'] = test_imputed_std['y']
# print('The C-Statistics is ',roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']))
# Outcomes_train['prob_bin'] = pd.qcut(Outcomes_train['probs'], q=20)
#
# Outcomes_train.groupby(['prob_bin'])['y'].sum()


# Full Model
train_and_val = dfsf.df_df_row_concat(train_imputed_std, val_imputed_std)

all_train = dfsf.df_df_row_concat(train_and_val, test_imputed_std)

final_logit = sm.Logit(all_train['y'], all_train[variables])

# fit the model
final_result = final_logit.fit()
final_result.summary()

# save the model to disk
model_name_new = 'prediction_pickles/model.pkl'
pickle.dump(final_result, open(model_name_new, 'wb'))
print("[INFO]: Finished saving model...")




