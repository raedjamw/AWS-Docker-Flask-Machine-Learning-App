import pickle
import pandas as pd
from flask import Flask, jsonify, request
import ML_Model.Test_Dataframes_pickles.dataframe_shape_functions as dfsf
import ML_Model.Test_Transformations_pickles.transformation_functions as tff

app = Flask(__name__)

model_path = "ML_Model/prediction_pickles/model.pkl"


@app.route("/predict", methods=['POST'])
def predict_model():
    test = request.get_json(force=True)

    test = pd.DataFrame.from_dict(test)

    # formatting to numeric
    test['x12'] = test['x12'].str.replace('$', '')
    test['x12'] = test['x12'].str.replace(',', '')
    test['x12'] = test['x12'].str.replace(')', '')
    test['x12'] = test['x12'].str.replace('(', '-')
    test['x12'] = test['x12'].astype(float)
    test['x63'] = test['x63'].str.replace('%', '')
    test['x63'] = test['x63'].astype(float)

    # call the imputer function
    test_imputed = tff.imputer_func(test, 'mean', ['x5', 'x31', 'x81', 'x82'])

    #  loading the train_columns from disk
    train_col_path = "ML_Model/prediction_pickles/train_columns.pkl"
    train_columns = pickle.load(open(train_col_path, 'rb'))


    #  loading the fitted_std_scaler from disk
    fitted_std_scaler_path = "ML_Model/prediction_pickles/fitted_std_scaler.pkl"
    fitted_std_scaler = pickle.load(open(fitted_std_scaler_path, 'rb'))

    # stadardize the numeric variables with standardscaler()
    test_imputed_std = pd.DataFrame(fitted_std_scaler.transform(test_imputed), columns=train_columns)

    # 3 convert nominal categorical variables to numeric via 1 hot encoding
    dumb5 = tff.dummies_funct(test['x5'], 'x5')
    test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb5)

    # 3 convert nominal categorical variables to numeric
    dumb31 = tff.dummies_funct(test['x31'], 'x31')
    test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb31)

    # 3 convert nominal categorical variables to numeric
    dumb81 = tff.dummies_funct(test['x81'], 'x81')
    test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb81)

    # 3 convert nominal categorical variables to numeric
    dumb82 = tff.dummies_funct(test['x82'], 'x82')
    test_imputed_std = dfsf.df_df_concat(test_imputed_std, dumb82)


    # Passing data to model & loading the model from disk
    model = pickle.load(open(model_path, 'rb'))

    # loading the model variables from disk
    variables_path = "ML_Model/prediction_pickles/variables.pkl"
    variables = pickle.load(open(variables_path, 'rb'))


    predictions = pd.DataFrame(model.predict(test_imputed_std[variables])).rename(columns={0: 'probs'})
    predictions['prob_bin'] = pd.qcut(predictions['probs'], q=20)

    # position of 75%
    upperq = int(20 * 0.75)

    API_Output = predictions

    # Lower bound of 75th %
    qt_75 = API_Output.prob_bin.cat.categories[upperq]

    qt_75 = qt_75.left

    # Create a column of all lower bounds and convert Caterory to floats
    API_Output['left'] = API_Output['prob_bin'].apply(lambda x: x.left).astype('float')

    # Re-label 75th% to 'event',below that label as 'no event'
    API_Output['left'] = ['event' if x >= qt_75 else 'no event' for x in API_Output['left']]

    API_Output.drop(columns=['prob_bin'], inplace=True)

    API_Output.rename(columns={'probs': 'phat', 'left': 'business_outcome'}, inplace=True)

    sort_vars = sorted(variables)
    sort_vars = pd.DataFrame(sort_vars).rename(columns={0: 'model_variables'})

    return jsonify(API_Output.to_dict(), sort_vars.to_dict())


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)



