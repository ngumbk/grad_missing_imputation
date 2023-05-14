import numpy as np
import pandas as pd
import math
from sklearn.metrics import f1_score, mean_squared_error


def make_num_metrics_matrix(dataset, name, mechanism, miss_percent, methods_map, var_map, count_mean=False, *sets):
    '''
    dataset - исходный датасет, с которым будут сравниваться восстановленные датасеты
    name - название исходного датасета
    mechanism - механизм пропуска, использованный в переданных наборах
    miss_percent - процент пропусков в переданном наборе
    methods_map - словарь с названиями функций импутации
    var_map - словарь с пользовательским определением типа каждой переменной
    count_mean=False - Если True, - считает среднее среди сетов с разными долями пропусков
    *sets - восстановленные наборы
    '''
    dataset = dataset.copy()
    c = dataset.shape[1]  # количество переменных в датасете
    num_columns = []

    # Заполняем num_columns
    for i in range(c):
        if var_map[dataset.columns[i]] == 'num':
            num_columns.append(dataset.columns[i])
    c = len(num_columns)

    # создаем список ключей для получаемых датафреймов
    keys_set = [f'{name}_{mechanism}{miss_percent}_{num_columns[i]}' for i in range(0, c)]
    matrix = pd.DataFrame(np.zeros((len(methods_map), c)), columns=num_columns)
    matrix = matrix.rename(index=methods_map)

    # Считаем метрику RMSE по столбцам
    for i in range(len(methods_map)):
        for j in range(len(keys_set)):
            y_pred = sets[i][keys_set[j]][num_columns[j]]
            y_true = dataset[num_columns[j]]
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            matrix.iloc[i, j] = rmse

    if count_mean:
        return None
    else:
        return matrix


def make_cat_metrics_matrix(dataset, name, mechanism, miss_percent, methods_map, var_map, count_mean=False, *sets):
    '''
    dataset - исходный датасет, с которым будут сравниваться восстановленные датасеты
    name - название исходного датасета
    mechanism - механизм пропуска, использованный в переданных наборах
    miss_percent - процент пропусков в переданном наборе
    methods_map - словарь с названиями функций импутации
    var_map - словарь с пользовательским определением типа каждой переменной
    count_mean=False - Если True, - считает среднее среди сетов с разными долями пропусков
    *sets - восстановленные наборы
    '''
    dataset = dataset.copy()
    c = dataset.shape[1]  # количество переменных в датасете
    cat_columns = []

    # Заполняем cat_columns
    for i in range(c):
        if var_map[dataset.columns[i]] != 'num':
            cat_columns.append(dataset.columns[i])
    c = len(cat_columns)

    # создаем список ключей для получаемых датафреймов
    keys_set = [f'{name}_{mechanism}{miss_percent}_{cat_columns[i]}' for i in range(0, c)]
    matrix = pd.DataFrame(np.zeros((len(methods_map), c)), columns=cat_columns)
    matrix = matrix.rename(index=methods_map)

    # Считаем метрику F1 по столбцам
    for i in range(len(methods_map)):
        for j in range(len(keys_set)):
            y_pred = sets[i][keys_set[j]][cat_columns[j]]
            y_true = dataset[cat_columns[j]]
            f1 = f1_score(y_true, y_pred, average='micro')
            matrix.iloc[i, j] = f1

    return matrix
