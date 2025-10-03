import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import re
import matplotlib.pyplot as plt

def trim_rows():
    random.seed(41)
    # If wanting the 200,000 rows for data analysis, set to 200,000. This number is to filter
    # out null values, which are statistically significant for analysis of the dataset but
    # difficult to use for model training.
    rows_selected = 450000

    original_data = pd.read_csv("used_cars_data.csv")
    original_rows = len(original_data)

    random_indexes = random.sample(range(original_rows), rows_selected)
    trimmed_rows_data = original_data.iloc[random_indexes]
    trimmed_rows_data.to_csv("trimmed_rows_used_cars_data.csv", index=False)

def trim_columns():
    trimmed_rows_data = pd.read_csv("trimmed_rows_used_cars_data.csv")
    # Delete columns that have mostly null values as well as the description and picture url columns,
    # since we want to focus on working with tabular data and not text or image data.
    # 'listed_date' is unnecessary since we already have 'daysonmarket'.
    # 'power' is unnecessary since it is just a different representation of horsepower.
    columns_to_delete = ['bed', 'bed_height', 'bed_length', 'cabin', 'combine_fuel_economy', 'vehicle_damage_category', 'wheel_system_display',
                         'is_certified', 'is_cpo', 'is_oemcpo', 'description', 'main_picture_url', 'major_options', 'transmission_display',
                         'listed_date', 'engine_type', 'dealer_zip', 'city', 'exterior_color', 'trimId', 'trim_name', 'make_name',
                         'vin', 'sp_name', 'model_name', 'frame_damaged', 'salvage', 'theft_title', 'listing_id', 'sp_id']
    
    trimmed_rows_data.drop(columns=columns_to_delete, inplace=True)
    trimmed_rows_data.to_csv("trimmed_used_cars_data.csv", index=False)

def group_colors(color):
    if isinstance(color, str):
        color = color.lower()
        if 'red' in color or 'scarlet' in color:
            return 'red'
        elif 'white' in color:
            return 'white'
        elif 'black' in color:
            return 'black'
        elif 'silver' in color:
            return 'silver'
        elif 'blue' in color:
            return 'blue'
        elif 'green' in color:
            return 'green'
        elif 'beige' in color:
            return 'beige'
        elif 'brown' in color:
            return 'brown'
        elif 'gray' in color or 'grey' or 'ash' in color:
            return 'gray'
        elif 'metallic' in color:
            return 'metallic'
        elif 'pearl' in color:
            return 'pearlcoat'
        elif 'leather' in color:
            return 'leather'
        else:
            return 'other'

def parse_out_numerical_value(textual_val):
    if isinstance(textual_val, str):
        matches = re.findall(r"\d+\.\d+|\d+", textual_val)
        if matches:
            return float(matches[0])
    return None

def parse_out_torque_hp(textual_val):
    if isinstance(textual_val, str):
        values = re.findall(r'(\d+(\.\d+)?) (lb-ft|hp) @ (\d+(,\d+)?) RPM', textual_val, re.IGNORECASE)
        if values:
            return values
    return None

def update_owner_count(row):
    if pd.isna(row['owner_count']):
        return 0 if row['is_new'] == 1 else 1
    else:
        return row['owner_count']

def preprocess_data():
    preprocessed_data = pd.read_csv("trimmed_used_cars_data.csv")

    preprocessed_data['general_interior_color'] = preprocessed_data['interior_color'].apply(group_colors)

    preprocessed_data.drop(['interior_color'], axis=1, inplace=True)

    preprocessed_data['height_num'] = preprocessed_data['height'].apply(parse_out_numerical_value)
    preprocessed_data['maximum_seating_num'] = preprocessed_data['maximum_seating'].apply(parse_out_numerical_value)
    preprocessed_data['front_legroom_num'] = preprocessed_data['front_legroom'].apply(parse_out_numerical_value)
    preprocessed_data['back_legroom_num'] = preprocessed_data['back_legroom'].apply(parse_out_numerical_value)
    preprocessed_data['fuel_tank_volume_num'] = preprocessed_data['fuel_tank_volume'].apply(parse_out_numerical_value)
    preprocessed_data['length_num'] = preprocessed_data['length'].apply(parse_out_numerical_value)
    preprocessed_data['wheelbase_num'] = preprocessed_data['wheelbase'].apply(parse_out_numerical_value)
    preprocessed_data['width_num'] = preprocessed_data['width'].apply(parse_out_numerical_value)

    preprocessed_data.drop(['height', 'maximum_seating', 'front_legroom', 'back_legroom', 'fuel_tank_volume',
                            'length', 'wheelbase', 'width'], axis=1, inplace=True)
    
    preprocessed_data['torque_rpm_vals'] = preprocessed_data['torque'].apply(parse_out_torque_hp)
    preprocessed_data['power_rpm_vals'] = preprocessed_data['power'].apply(parse_out_torque_hp)

    preprocessed_data['torque_foot_pound'] = preprocessed_data['torque_rpm_vals'].apply(lambda x: float(x[0][0]) if x is not None else None)
    preprocessed_data['torque_rpm'] = preprocessed_data['torque_rpm_vals'].apply(lambda x: int(x[0][3].replace(',', '')) if x is not None else None)
    preprocessed_data['power_hp'] = preprocessed_data['power_rpm_vals'].apply(lambda x: float(x[0][0]) if x is not None else None)
    preprocessed_data['power_rpm'] = preprocessed_data['power_rpm_vals'].apply(lambda x: int(x[0][3].replace(',', '')) if x is not None else None)

    preprocessed_data.drop(['torque', 'power', 'torque_rpm_vals', 'power_rpm_vals'], axis=1, inplace=True)

    binary_encoding_cols = ['fleet', 'franchise_dealer', 'has_accidents', 'isCab', 'is_new']
    preprocessed_data[binary_encoding_cols] = preprocessed_data[binary_encoding_cols].fillna(0).astype(int)

    preprocessed_data['owner_count'] = preprocessed_data.apply(update_owner_count, axis=1)
    preprocessed_data = preprocessed_data.dropna()

    preprocessed_data.drop(preprocessed_data.tail(38243).index, inplace=True)

    one_hot_encoding_cols = ['body_type', 'engine_cylinders', 'franchise_make', 'fuel_type', 'listing_color',
                             'transmission', 'wheel_system', 'general_interior_color']
    preprocessed_data = pd.get_dummies(preprocessed_data, columns=one_hot_encoding_cols, prefix=one_hot_encoding_cols)

    scaler = StandardScaler()

    cols = preprocessed_data.columns[preprocessed_data.columns != 'price']
    preprocessed_data[cols] = scaler.fit_transform(preprocessed_data[cols])
    
    preprocessed_data.to_csv("preprocessed_data.csv", index=False)

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2) Score:", r2)
    print("Mean Absolute Error (MAE):", mae)

def single_linear_regression(feature_engineering=False, fit_intercept=True):
    data = pd.read_csv("preprocessed_data.csv")
    target_variable = "price"

    feature_groups = ['city_fuel_economy', 'daysonmarket', 'engine_displacement', 'fleet',
                      'franchise_dealer', 'has_accidents', 'highway_fuel_economy', 'horsepower',
                      'isCab', 'is_new', 'latitude', 'longitude', 'mileage', 'owner_count',
                      'savings_amount', 'seller_rating', 'year', 'height_num', 'maximum_seating_num',
                      'front_legroom_num', 'back_legroom_num', 'fuel_tank_volume_num', 'length_num',
                      'wheelbase_num', 'width_num', 'torque', 'power', 'body_type', 'engine_cylinders', 
                      'franchise_make', 'fuel_type', 'listing_color', 'transmission', 'wheel_system', 'general_interior_color']

    if feature_engineering:

        r2_values = {}

        for feature_group in feature_groups:
            features = [col for col in data.columns if col.startswith(feature_group)]
            X_group = data[features]
            y = data[target_variable]

            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_group, y)

            y_pred = model.predict(X_group)
            r2 = r2_score(y, y_pred)

            r2_values[feature_group] = r2
        
        plt.bar(r2_values.keys(), r2_values.values())
        plt.ylabel("R-squared")
        plt.title("R-squared Values for Feature Groups")
        plt.xticks(rotation=45, ha="right")
        plt.show()
    
    else:
        feature = [col for col in data.columns if col.startswith('horsepower')]
        X = data[feature]
        y = data[target_variable]
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)

        y_pred = model.predict(X)

        evaluate_model(y_test=y, y_pred=y_pred)

def multiple_linear_regression(prefix_list, fit_intercept=True):
    data = pd.read_csv("preprocessed_data.csv")
    X = data.drop((col for col in data.columns if not any(col.startswith(prefix) for prefix in prefix_list)), axis=1)
    y = data["price"] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(y_test=y_test, y_pred=y_pred)

def knn_regression(prefix_list, k=3):
    data = pd.read_csv('preprocessed_data.csv')
    target_variable = "price"
    X = data.drop((col for col in data.columns if not any(col.startswith(prefix) for prefix in prefix_list)), axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    model = KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(y_test=y_test, y_pred=y_pred)

def regression_tree(prefix_list, max_depth=10):
    data = pd.read_csv("preprocessed_data.csv")

    target_variable = "price"
    X = data.drop((col for col in data.columns if not any(col.startswith(prefix) for prefix in prefix_list)), axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    model = DecisionTreeRegressor(max_depth=max_depth)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(y_test=y_test, y_pred=y_pred)

def random_forest_regression(prefix_list, n_estimators=10, max_depth=10):
    data = pd.read_csv("preprocessed_data.csv")
    target_variable = "price"
    X = data.drop((col for col in data.columns if not any(col.startswith(prefix) for prefix in prefix_list)), axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=41) 

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(y_test=y_test, y_pred=y_pred)

def vanilla_models():
    print("Vanilla multiple linear regression")
    multiple_linear_regression(prefix_list=all_feature_groups)
    print("Vanilla k-nearest-neighbors regression")
    knn_regression(prefix_list=all_feature_groups)
    print("Vanilla regression tree")
    regression_tree(prefix_list=all_feature_groups)
    print("Vanilla random forest regression")
    random_forest_regression(prefix_list=all_feature_groups)
    print("Vanilla simple regression")
    single_linear_regression()

def cross_validate_models():
    print("Cross validating models with 5 folds")
    print("Cross validating multiple linear regression")
    cross_validate_single_model(LinearRegression())
    print("Cross validating k-nearest-neighbors regression")
    cross_validate_single_model(KNeighborsRegressor(n_neighbors=3))
    print("Cross validating regression tree")
    cross_validate_single_model(DecisionTreeRegressor(max_depth=10))
    print("Cross validating random forest regression")
    cross_validate_single_model(RandomForestRegressor(n_estimators=10, max_depth=10, random_state=41))
    print("Cross validating simple linear regression")
    cross_validate_single_model(LinearRegression(), simple_linear_regression=True)


def cross_validate_single_model(model, folds=5, simple_linear_regression = False):
    data = pd.read_csv("preprocessed_data.csv")
    X = data.drop("price", axis=1)
    if simple_linear_regression:
        feature = [col for col in data.columns if col.startswith('horsepower')]
        X = data[feature]
    y = data["price"] 

    kf = KFold(n_splits=folds, shuffle=True)
    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    scoring = {
        'r2': 'r2',
        'neg_rmse': rmse_scorer,
        'neg_mae': mae_scorer,
    }

    cv_result = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)

    mean_r2 = cv_result['test_r2'].mean()
    mean_rmse = -cv_result['test_neg_rmse'].mean()
    mean_mae = -cv_result['test_neg_mae'].mean()

    print("Mean R-squared Score:", mean_r2)
    print("Mean Root Mean Squared Error:", mean_rmse)
    print("Mean Mean Absolute Error:", mean_mae)

    print("Mean Mean Squared Error:", mean_rmse * mean_rmse)

def feature_engineering():

    print("Performing feature engineering with multiple linear regression")
    print("All feature groups:")
    multiple_linear_regression(prefix_list=all_feature_groups)
    print("Most feature groups:")
    multiple_linear_regression(prefix_list=most_feature_groups)
    print("Best feature groups:")
    multiple_linear_regression(prefix_list=best_feature_groups)

    print("Performing feature engineering with k-nearest-neighbors regression")
    print("Most feature groups:")
    knn_regression(prefix_list=most_feature_groups)
    print("Best feature groups:")
    knn_regression(prefix_list=best_feature_groups)

    print("Performing feature engineering with regression tree")
    print("Most feature groups:")
    regression_tree(prefix_list=most_feature_groups)
    print("Best feature groups:")
    regression_tree(prefix_list=best_feature_groups)

    print("Performing feature engineering with random forest regression")
    print("Most feature groups:")
    random_forest_regression(prefix_list=most_feature_groups)
    print("Best feature groups:")
    random_forest_regression(prefix_list=best_feature_groups)

    print("Performing feature engineering with simple linear regression")
    single_linear_regression(feature_engineering=True)

def hyper_parameter_tuning():

    print("Performing hyperparameter tuning with multiple linear regression")
    print("Not calculating fit intercept")
    multiple_linear_regression(prefix_list=all_feature_groups, fit_intercept=False)
    
    print("Performing hyperparameter tuning with k-nearest-neighbor regression")
    print("Number of nearest neighbors = 1")
    knn_regression(prefix_list=all_feature_groups, k=1)
    print("Number of nearest neighbors = 5")
    knn_regression(prefix_list=all_feature_groups, k=5)
    print("Number of nearest neighbors = 10")
    knn_regression(prefix_list=all_feature_groups, k=10)

    print("Performing hyperparameter tuning with regression tree")
    print("Max depth = 5")
    regression_tree(prefix_list=all_feature_groups, max_depth=5)
    print("Max depth = 25")
    regression_tree(prefix_list=all_feature_groups, max_depth=25)
    print("Max depth = 50")
    regression_tree(prefix_list=all_feature_groups, max_depth=50)

    print("Performing hyperparameter tuning with random forest regression")
    print("Number of estimators = 3")
    random_forest_regression(prefix_list=all_feature_groups, n_estimators=5)
    print("Number of estimators = 20")
    random_forest_regression(prefix_list=all_feature_groups, n_estimators=20)
    print("Number of estimators = 50")
    random_forest_regression(prefix_list=all_feature_groups, n_estimators=50)

    print("Performing hyperparameter tuning with simple linear regression")
    print("Not calculating fit intercept")
    single_linear_regression(fit_intercept=False)

def run_all_models():
    vanilla_models()
    cross_validate_models()
    feature_engineering()
    hyper_parameter_tuning()

    

all_feature_groups = ['city_fuel_economy', 'daysonmarket', 'engine_displacement', 'fleet',
                      'franchise_dealer', 'has_accidents', 'highway_fuel_economy', 'horsepower',
                      'isCab', 'is_new', 'latitude', 'longitude', 'mileage', 'owner_count',
                      'savings_amount', 'seller_rating', 'year', 'height_num', 'maximum_seating_num',
                      'front_legroom_num', 'back_legroom_num', 'fuel_tank_volume_num', 'length_num',
                      'wheelbase_num', 'width_num', 'torque', 'power', 'body_type', 'engine_cylinders', 
                      'franchise_make', 'fuel_type', 'listing_color', 'transmission', 'wheel_system', 'general_interior_color']

most_feature_groups = ['city_fuel_economy', 'engine_displacement',
                      'highway_fuel_economy', 'horsepower',
                      'is_new', 'mileage', 'owner_count',
                      'year', 'height_num', 'maximum_seating_num',
                      'back_legroom_num', 'fuel_tank_volume_num', 'length_num',
                      'wheelbase_num', 'width_num', 'torque', 'power', 'body_type', 'engine_cylinders', 
                      'franchise_make', 'transmission', 'wheel_system']
best_feature_groups = ['city_fuel_economy', 'engine_displacement',
                      'highway_fuel_economy', 'horsepower',
                      'fuel_tank_volume_num', 'length_num',
                      'wheelbase_num', 'torque', 'power', 'engine_cylinders', 
                      'franchise_make','wheel_system']

#trim_rows()
#trim_columns()
#preprocess_data()
#multiple_linear_regression()
#multiple_linear_regression_cross_val(10)
#polynomial_regression()
#knn_regression()
#svm_regression()
#regression_tree()
#random_forest_regression()
#feature_engineering()
#single_linear_regression()
#hyper_parameter_tuning()
#cross_validate_models()
run_all_models()