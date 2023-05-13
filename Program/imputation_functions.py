import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import Pool, CatBoostClassifier, CatBoostRegressor


def impute_average(dataset, column, mode='median', verbose=True):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    column - столбец, подлежащий восстановлению
    mode='median' - значение, которым восстанавливается столбец, также есть варианты mean, mode
    standard_deviation=False - добавление случайных чисел в пределах стандартного отклонения в массив импутируемых значений
    verbose=True - флаг вывода, если True - то функция будет выводить сообщения на экран
    '''
    dataset = dataset.copy()
    dataset_col = dataset.loc[:, column]
    if mode == 'median':
        imputed_value = dataset_col.median()
    elif mode == 'mean':
        imputed_value = round(dataset_col.mean(), 1)
    elif mode == 'mode':
        imputed_value = dataset_col.mode().iloc[0]

    if verbose:
        print('Значение для импутации:', imputed_value)

    predict_vector = np.array([imputed_value for i in range(dataset[column].isna().sum())])

    dataset[column] = dataset[column].fillna(value=imputed_value)

    return dataset


def impute_linreg(dataset, column, verbose=True):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    column - столбец, подлежащий восстановлению
    verbose=True - флаг вывода, если True - то функция будет выводить сообщения на экран
    '''
    dataset = dataset.copy()
    mask_notnull = dataset[column].notnull()
    full_data = dataset.loc[mask_notnull == True, :]  # берем только те строки, в которых в column нет пропуска
    y = np.array(full_data.loc[:, column]).reshape(-1, 1)

    x = np.array(full_data.loc[:, full_data.columns != column])

    missing_data = dataset.loc[mask_notnull == False, :]
    x_pred = np.array(missing_data.loc[:, full_data.columns != column])

    # На случай, если разберусь с excluded_columns
    # x_pred = np.array(missing_data.loc[:, ~full_data.columns.isin(excluded_columns)])

    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    x_pred = scaler.fit_transform(x_pred)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    if verbose:
        print('Показатели модели:', reg.coef_, reg.intercept_, sep='\n', end='\n\n')
        print(reg.score(x_train, y_train), reg.score(x_test, y_test), sep='\n')

    predict_vector = reg.predict(x_pred)

    # кривой способ замены вместо pd.fillna()
    miss_n = 0
    for i in range(dataset.shape[0]):
        if pd.isna(dataset.loc[i, column]):
            dataset.loc[i, column] = predict_vector[miss_n]
            miss_n += 1

    return dataset


def impute_knn(dataset, column, verbose=True):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    column - столбец, подлежащий восстановлению
    verbose=True - флаг вывода, если True - то функция будет выводить сообщения на экран
    '''
    dataset = dataset.copy()  # копируем датасет чтобы не вносить изменения по ссылке
    mask_notnull = dataset[
        column].notnull()  # создаем маску, с True на тех позициях, где в dataset[column] нет пропуска
    full_data = dataset.loc[mask_notnull == True,
                :]  # применяем маску чтобы получить срез датасета для обучения и теста
    y = np.array(full_data.loc[:, column]).reshape(-1, 1)  # берем таргеты из датасета для обучения

    x = np.array(full_data.loc[:, full_data.columns != column])  # составляем массив данных для обучения

    missing_data = dataset.loc[mask_notnull == False, :]  # срез датасета для восстановления пропусков в dataset
    x_pred = np.array(
        missing_data.loc[:, full_data.columns != column])  # массив признаков для подстановки в обученную модель

    # Нормализация X
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_pred = scaler.fit_transform(x_pred)

    # Деление выборки на обучающую и тестовую
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                     leaf_size=30, p=2, metric='minkowski', n_jobs=-1)

    # "Обучение" модели
    knn_model.fit(x_train, y_train.ravel())

    if verbose:
        print(knn_model.score(x_train, y_train), knn_model.score(x_test, y_test), sep='\n')

    predict_vector = knn_model.predict(x_pred)

    # кривой способ замены pd.fillna()
    miss_n = 0
    for i in range(dataset.shape[0]):
        if pd.isna(dataset.loc[i, column]):
            dataset.loc[i, column] = predict_vector[miss_n]
            miss_n += 1

    return dataset


def impute_catboost_cat(dataset, column, var_map, catboost_params={}, verbose=100):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    column - столбец, подлежащий восстановлению
    var_map - словарь, спользуется для определения типа классификации и выбора лосс-функции
    catboost_params={} - список параметров catboost
    verbose=100 - флаг вывода, если True - то функция будет выводить сообщения на экран
    '''

    dataset = dataset.copy()
    mask_notnull = dataset[column].notnull()
    full_data = dataset.loc[mask_notnull == True, :]  # берем только те строки, в которых в column нет пропуска
    y = np.array(full_data.loc[:, column]).reshape(-1, 1)

    x = np.array(full_data.loc[:, full_data.columns != column])

    missing_data = dataset.loc[mask_notnull == False, :]
    x_pred = np.array(missing_data.loc[:, full_data.columns != column])

    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    x_pred = scaler.fit_transform(x_pred)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    learn_pool = Pool(
        x_train,
        y_train
    )
    test_pool = Pool(
        x_test,
        y_test
    )

    catboost_default_params = {
        'iterations': 1000,
        'learning_rate': 0.01,
        'eval_metric': 'TotalF1',
        'leaf_estimation_method': 'Gradient',
        'bootstrap_type': 'Bernoulli'
    }

    if var_map[column] == 'cat':
        catboost_default_params['objective'] = 'MultiClass'
    else:
        catboost_default_params['objective'] = 'CrossEntropy'

    catboost_default_params.update(catboost_params)

    catboost = CatBoostClassifier(**catboost_default_params)
    catboost.fit(learn_pool, eval_set=test_pool, verbose=verbose)

    predict_vector = catboost.predict(x_pred)

    # косая замена pd.fillna()
    miss_n = 0
    for i in range(dataset.shape[0]):
        if pd.isna(dataset.loc[i, column]):
            dataset.loc[i, column] = predict_vector[miss_n]
            miss_n += 1

    return dataset


def impute_catboost_num(dataset, column, catboost_params={}, verbose=100):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    column - столбец, подлежащий восстановлению
    excluded_columns=[] - список колонок, не используемых в обучении
    catboost_params={} - список параметров catboost
    verbose=100 - флаг вывода, если True - то функция будет выводить сообщения на экран
    '''

    dataset = dataset.copy()
    mask_notnull = dataset[column].notnull()
    full_data = dataset.loc[mask_notnull == True, :]  # берем только те строки, в которых в column нет пропуска
    y = np.array(full_data.loc[:, column]).reshape(-1, 1)

    x = np.array(full_data.loc[:, full_data.columns != column])

    missing_data = dataset.loc[mask_notnull == False, :]
    x_pred = np.array(missing_data.loc[:, full_data.columns != column])

    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    x_pred = scaler.fit_transform(x_pred)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    learn_pool = Pool(
        x_train,
        y_train
    )
    test_pool = Pool(
        x_test,
        y_test
    )

    catboost_default_params = {
        'iterations': 1000,
        'learning_rate': 0.01,
        'depth': 2,
        'leaf_estimation_method': 'Gradient',
        'bootstrap_type': 'Bernoulli',
        'objective': 'RMSE',
    }

    catboost_default_params.update(catboost_params)

    catboost = CatBoostRegressor(**catboost_default_params)
    catboost.fit(learn_pool, eval_set=test_pool, verbose=verbose)

    predict_vector = catboost.predict(x_pred)

    # косая замена pd.fillna()
    miss_n = 0
    for i in range(dataset.shape[0]):
        if pd.isna(dataset.loc[i, column]):
            dataset.loc[i, column] = predict_vector[miss_n]
            miss_n += 1

    return dataset


@st.cache_data
def impute_set(df_sets, var_map, cat_impute_function, num_impute_function, dataset, mechanism, name,
               miss_parameters=[5, 15, 25], avg_impute_mode='median'):
    '''
    df_sets - наборы датасетов в виде массива словарей 3x14, где каждое значение словаря - датасет с пропусками в соответствие с кодировкой ключа
    var_map - словарь с пользовательским определением типа каждой переменной
    cat_impute_function - функция, с помощью которой востанавливаются категориальные переменные набора
    num_impute_function - функция, с помощью которой востанавливается численные переменные набора
    dataset - исходный набор данных в виде pandas dataframe
    mechanism - механизм пропусков, допускаемых в данном наборе датасетов. Может принимать значение: MCAR, MAR или MNAR
    name - название ключей словаря с набором датасетов
    miss_parameters=[5, 15, 25] - список параметров пропуска, принимает 3 параметра, означающих примерные доли пропусков в результирующих наборах
    avg_impute_mode='median' - способ восстановления при использования функции восстановления средним ['median', 'mean', 'mode']
    '''
    # 4 параметра нужны по сути для составления ключа

    imputed_df_sets = []  # список наборов, который будет возвращен
    c = len(df_sets[0])  # количество переменных в каждом датасете

    for i in range(3):
        df_set = df_sets[i]  # Берем один набор (словарь, содержащий 14 df) из 3
        imputed_df_set = {}  # Создаем словарь для ВОССТАНОВЛЕННЫХ ДАТАФРЕЙМОВ (нужно для целостности)

        keys_set = [f'{name}_{mechanism}{miss_parameters[i]}_{dataset.columns[j]}' for j in
                    range(c)]  # создаем список ключей для получаемых датафреймов
        for j in range(c):  # восстанавливаем каждый датасет
            print(str(i) + ':' + str(j), keys_set[j], dataset.columns[j], df_set[keys_set[j]], sep='\n')
            if (var_map[dataset.columns[j]] == 'cat' or var_map[dataset.columns[
                j]] == 'bin') and cat_impute_function == impute_average:  # особые случаи для функции восстановления средним
                imputed_df_set[keys_set[j]] = cat_impute_function(df_set[keys_set[j]], dataset.columns[j], 'mode',
                                                                  verbose=False)  # Кейс средней импутации для кат.
            elif var_map[dataset.columns[j]] == 'num' and num_impute_function == impute_average:
                imputed_df_set[keys_set[j]] = num_impute_function(df_set[keys_set[j]], dataset.columns[j],
                                                                  avg_impute_mode,
                                                                  verbose=False)  # Кейс средней импутации для числ.
            elif (var_map[dataset.columns[j]] == 'cat' or var_map[
                dataset.columns[j]] == 'bin') and cat_impute_function == impute_catboost_cat:
                imputed_df_set[keys_set[j]] = cat_impute_function(df_set[keys_set[j]], dataset.columns[j], var_map,
                                                                  verbose=False)  # кейс передачи var_map
            elif var_map[dataset.columns[j]] == 'cat' or var_map[dataset.columns[j]] == 'bin':
                imputed_df_set[keys_set[j]] = cat_impute_function(df_set[keys_set[j]], dataset.columns[j],
                                                                  verbose=False)
            elif var_map[dataset.columns[j]] == 'num':
                imputed_df_set[keys_set[j]] = num_impute_function(df_set[keys_set[j]], dataset.columns[j],
                                                                  verbose=False)
        imputed_df_sets.append(imputed_df_set)

    return imputed_df_sets
