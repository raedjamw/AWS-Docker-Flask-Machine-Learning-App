# -*- coding: utf-8 -*-
# std libraries
import numpy as np
import pandas as pd
import sys
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from collections import Counter

import Test_Dataframes_pickles.dataframe_shape_functions as dfsf
import Test_Transformations_pickles.transformation_functions as tff


# inspect package versions
print("python version " + sys.version)

# function to get csv as dataframe
raw_train = dfsf.df_from_csv(r'exercise_26_train.csv')
raw_test = dfsf.df_from_csv(r'exercise_26_test.csv')


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

# Feature Engineering

train_val = raw_train.copy(deep=True)

#1 Formatting to numeric
train_val['x12'] = train_val['x12'].str.replace('$','')
train_val['x12'] = train_val['x12'].str.replace(',','')
train_val['x12'] = train_val['x12'].str.replace(')','')
train_val['x12'] = train_val['x12'].str.replace('(','-')
train_val['x12'] = train_val['x12'].astype(float)
train_val['x63'] = train_val['x63'].str.replace('%','')
train_val['x63'] = train_val['x63'].astype(float)


# train test split the Trining data
x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)


# 2. smashing sets back together
train = dfsf.series_df_concat(x_train, y_train,True)
val = dfsf.series_df_concat(x_val, y_val, True)
test = dfsf.series_df_concat(x_test, y_test, True)

# 3. With mean imputation from Train set
train_imputed = tff.imputer_func(train,'mean',['y', 'x5', 'x31',  'x81' ,'x82'])
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

# Saving the train columns to disk
train_columns = train_imputed.columns
print(train_columns)
train_col_fl = 'prediction_pickles/train_columns.pkl'
pickle.dump(train_columns, open(train_col_fl, 'wb'))
print("[INFO]: Finished saving train_columns...")

# pickle std scaler function
std_scaler_fl = 'prediction_pickles/fitted_std_scaler.pkl'
pickle.dump(std_scaler, open(std_scaler_fl, 'wb'))
print("[INFO]: Finished saving fitted_std_scaler...")


# convert nominal categorical variables to numeric via 1 hot encoding
dumb5 = tff.dummies_funct(train['x5'],'x5')
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb5)

dumb31 = tff.dummies_funct(train['x31'],'x31')
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb31)

dumb81 = tff.dummies_funct(train['x81'],'x81')
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb81)

dumb82 = tff.dummies_funct(train['x82'],'x82')
train_imputed_std = dfsf.df_df_concat(train_imputed_std, dumb82)

train_imputed_std = dfsf.series_df_concat(train_imputed_std, train['y'],True)


# Initial Feature Selection

exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')

# Preliminary Model
"""
Starting with the train set. The L1 process creates biased parameter estimates.  
As a result, we will build a final model without biased estimators.
"""

variables = var_reduced['name'].to_list()
logit = sm.Logit(train_imputed_std['y'], train_imputed_std[variables])
# fit the model
result = logit.fit()
result.summary()


# Saving the variables to disk
variables_fl = 'prediction_pickles/variables.pkl'
pickle.dump(variables, open(variables_fl, 'wb'))
print("[INFO]: Finished saving variables...")


# save the model to disk
model_name = 'prediction_pickles/model.pkl'
pickle.dump(result, open(model_name, 'wb'))
print("[INFO]: Finished saving model...")



# Prepping the validation set

val_imputed = tff.imputer_func(val, 'mean', ['y', 'x5', 'x31',  'x81' ,'x82'])

val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=train_imputed.columns)

# convert nominal categorical variables to numeric via 1 hot encoding
dumb5 = tff.dummies_funct(val['x5'],'x5')
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb5)

dumb31 = tff.dummies_funct(val['x31'],'x31')
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb31)

dumb81 = tff.dummies_funct(val['x81'],'x81')
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb81)

dumb82 = tff.dummies_funct(val['x82'],'x82')
val_imputed_std = dfsf.df_df_concat(val_imputed_std, dumb82)
val_imputed_std  = dfsf.series_df_concat(val_imputed_std, val['y'],True)


# Prepping the test set

test_imputed = tff.imputer_func(test,'mean',['y', 'x5', 'x31',  'x81' ,'x82'])

test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=train_imputed.columns)

# # convert nominal categorical variables to numeric via 1 hot encoding
dumb5 = tff.dummies_funct(test['x5'],'x5')
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb5)

dumb31 = tff.dummies_funct(test['x31'],'x31')

test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb31)

dumb81 = tff.dummies_funct(test['x81'],'x81')

test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb81)

dumb82 = tff.dummies_funct(test['x82'],'x82')
test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb82)
test_imputed_std = dfsf.series_df_concat(test_imputed_std, test['y'],True)

# Models Results with the train set
Outcomes_train = pd.DataFrame(result.predict(train_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_train['y'] = train_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_train['y'], Outcomes_train['probs']))

# Models Results with the val set
Outcomes_val = pd.DataFrame(result.predict(val_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_val['y'] = val_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_val['y'], Outcomes_val['probs']))


# Models Results with the test set
Outcomes_test = pd.DataFrame(result.predict(test_imputed_std[variables])).rename(columns={0:'probs'})
Outcomes_test['y'] = test_imputed_std['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_test['y'], Outcomes_test['probs']))
Outcomes_train['prob_bin'] = pd.qcut(Outcomes_train['probs'], q=20)

Outcomes_train.groupby(['prob_bin'])['y'].sum()


# Full Model
train_and_val = dfsf.df_df_row_concat(train_imputed_std, val_imputed_std)

all_train = dfsf.df_df_row_concat(train_and_val, test_imputed_std)

variables = var_reduced['name'].to_list()
final_logit = sm.Logit(all_train['y'], all_train[variables])

# fit the model
final_result = final_logit.fit()
final_result.summary()


Outcomes_train_final = pd.DataFrame(result.predict(all_train[variables])).rename(columns={0:'probs'})
Outcomes_train_final['y'] = all_train['y']
print('The C-Statistics is ',roc_auc_score(Outcomes_train_final['y'], Outcomes_train_final['probs']))
Outcomes_train_final['prob_bin'] = pd.qcut(Outcomes_train_final['probs'], q=20)
Outcomes_train_final.groupby(['prob_bin',])['y'].sum()


# Final Results for API
API_Output = Outcomes_train_final.copy()

# no of intervals
no_catgories = len(Outcomes_train_final.groupby(['prob_bin',])['y'].sum())
print(no_catgories)
# position of 75%
upperq = int(no_catgories*0.75)

# Lower bound of 75th %
qt_75 = API_Output.prob_bin.cat.categories[upperq]
qt_75= qt_75.left

# Create a column of all lower bounds and convert Caterory to floats
API_Output['left'] = API_Output['prob_bin'].apply(lambda x: x.left).astype('float')

# Re-label 75th% to 'event',below that label as 'no event'
API_Output['left'] = ['event' if x >= qt_75 else 'no event' for x in API_Output['left']]


API_Output.drop(columns=['y','prob_bin'],inplace=True)

API_Output.rename(columns={'probs':'phat','left':'business_outcome'},inplace=True)

# Model Variables for output
Model_Variables = variables
Model_Variables.sort()

