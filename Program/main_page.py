import markdown
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from make_misses import *
from imputation_functions import *
from impute_set_function import *
from make_metrics_functions import *


def main():
    # Программа работает ТОЛЬКО с датасетом CCHD
    cchd_data = pd.read_csv('../../ВКР/Datasets/processed_cleveland.csv')
    cchd_data.head(6)

    print(len(cchd_data.loc[cchd_data['ca'] == '?', 'ca']), len(cchd_data.loc[cchd_data['thal'] == '?', 'thal']),
          sep='\n')
    cchd_data.loc[cchd_data['ca'] == '?', 'ca'] = cchd_data['ca'].mode().iloc[0]
    cchd_data.loc[cchd_data['thal'] == '?', 'thal'] = cchd_data['thal'].mode().iloc[0]
    cchd_data = cchd_data.astype({'ca': 'int64', 'thal': 'int64'})
    print(cchd_data.shape)

    # 1.3
    cchd_data['cp'] = cchd_data['cp'].replace({2: 1, 3: 2, 4: 3})
    cchd_data['restecg'] = cchd_data['restecg'].replace({2: 1})
    cchd_data = cchd_data.drop(columns=['fbs'])
    cchd_data['slope'] = cchd_data['slope'].replace({1: 0, 2: 1, 3: 1})
    cchd_data['ca'] = cchd_data['ca'].replace({3: 2})
    cchd_data['thal'] = cchd_data['thal'].replace({7: 1, 6: 1, 3: 0})
    cchd_data['num'] = cchd_data['num'].replace({4: 3})

    cchd_var_map = {'age': 'num',
                    'sex': 'bin',
                    'cp': 'cat',
                    'trestbps': 'num',
                    'chol': 'num',
                    'restecg': 'bin',
                    'thalach': 'num',
                    'exang': 'bin',
                    'oldpeak': 'num',
                    'slope': 'bin',
                    'ca': 'cat',
                    'thal': 'bin',
                    'num': 'cat'}

    # 2
    # Проверка
    print(make_mcar(cchd_data, area='All', miss_percent=25))
    print(make_mar(cchd_data, area='sex', area_dependent='oldpeak', miss_parameter=30))
    print(make_mnar(cchd_data, area='oldpeak', ascending=False, miss_parameter=49))

    # 2.4
    # Проверка
    test_set = make_set_of_datasets(cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
    print(test_set[2]['cchd_data_MCAR25_age'])

    # 3
    # Создание 3 тестовых датасетов
    cchd_mar_num15_test = make_mar(cchd_data, area='thalach', area_dependent='age', miss_parameter=15)
    # вектор истинных значений
    num_test_mask_null = cchd_mar_num15_test['thalach'].isnull()
    num_true = np.array(cchd_data.loc[num_test_mask_null == True, 'thalach']).reshape(-1, 1)
    print('num_true.shape:', num_true.shape, end='\n\n')

    cchd_mar_cat15_b_test = make_mar(cchd_data, area='exang', area_dependent='thalach', miss_parameter=15)
    # вектор истинных значений
    cat_b_test_mask_null = cchd_mar_cat15_b_test['exang'].isnull()
    cat_b_true = np.array(cchd_data.loc[cat_b_test_mask_null == True, 'exang']).reshape(-1, 1)
    print('cat_b_true.shape:', cat_b_true.shape, end='\n\n')

    cchd_mar_cat15_m_test = make_mar(cchd_data, area='cp', area_dependent='thalach', miss_parameter=15)
    # вектор истинных значений
    cat_m_test_mask_null = cchd_mar_cat15_m_test['cp'].isnull()
    cat_m_true = np.array(cchd_data.loc[cat_m_test_mask_null == True, 'cp']).reshape(-1, 1)
    print('cat_m_true.shape:', cat_m_true.shape)

    # Восстанавливаем численную переменную медианой
    df = impute_average(cchd_mar_num15_test, 'thalach', mode='median')
    # Восстанавливаем численную переменную арифметическим средним
    df = impute_average(cchd_mar_num15_test, 'thalach', mode='mean')
    # Восстанавливаем бинарную категориальную переменную модой
    df = impute_average(cchd_mar_cat15_b_test, 'exang', mode='mode')
    # Восстанавливаем категориальную переменную модой
    df = impute_average(cchd_mar_cat15_m_test, 'cp', mode='mode')

    # ML shit
    df = impute_linreg(cchd_mar_num15_test, 'thalach')

    df = impute_knn(cchd_mar_cat15_b_test, 'exang')
    df = impute_knn(cchd_mar_cat15_m_test, 'cp')

    df = impute_catboost_cat(cchd_mar_cat15_b_test, 'exang', cchd_var_map)
    df = impute_catboost_cat(cchd_mar_cat15_m_test, 'cp', cchd_var_map)

    df = impute_catboost_num(cchd_mar_num15_test, 'thalach')

    # 4
    cchd_data_set_MCAR = make_set_of_datasets(cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_data_set_MAR = make_set_of_datasets(cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_data_set_MNAR = make_set_of_datasets(cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25])

    # 5
    dfs = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_average, impute_average,
                     cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25], avg_impute_mode='mode')

    cchd_MCAR_avg_avg = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_average, impute_average,
                                   cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25],
                                   avg_impute_mode='median')
    cchd_MCAR_knn_linreg = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_knn, impute_linreg,
                                      cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_MCAR_cat_cat = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_catboost_cat, impute_catboost_num,
                                   cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_MAR_avg_avg = impute_set(cchd_data_set_MAR, cchd_var_map, impute_average, impute_average,
                                  cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25], avg_impute_mode='median')
    cchd_MAR_knn_linreg = impute_set(cchd_data_set_MAR, cchd_var_map, impute_knn, impute_linreg,
                                     cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_MAR_cat_cat = impute_set(cchd_data_set_MAR, cchd_var_map, impute_catboost_cat, impute_catboost_num,
                                  cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_MNAR_avg_avg = impute_set(cchd_data_set_MNAR, cchd_var_map, impute_average, impute_average,
                                   cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25],
                                   avg_impute_mode='median')
    cchd_MNAR_knn_linreg = impute_set(cchd_data_set_MNAR, cchd_var_map, impute_knn, impute_linreg,
                                      cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_MNAR_cat_cat = impute_set(cchd_data_set_MNAR, cchd_var_map, impute_catboost_cat, impute_catboost_num,
                                   cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25])

    # 6
    num_methods_map = {0: 'Median Imputation', 1: 'Linear Regression', 2: 'Boosted Decision Tree'}
    metrics_MCAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, num_methods_map, cchd_var_map, False,
                                                cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
    print(metrics_MCAR5_num)

    cat_methods_map = {0: 'Mode Imputation', 1: 'K-Nearest Neighbours', 2: 'Boosted Decision Tree'}
    metrics_MCAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, cat_methods_map, cchd_var_map, False,
                                                cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
    print(metrics_MCAR5_cat)

    # 6.1.3
    # Числ.
    metrics_MCAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, num_methods_map, cchd_var_map, False,
                                                cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
    metrics_MCAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 15, num_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MCAR_avg_avg[1], cchd_MCAR_knn_linreg[1], cchd_MCAR_cat_cat[1])
    metrics_MCAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 25, num_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MCAR_avg_avg[2], cchd_MCAR_knn_linreg[2], cchd_MCAR_cat_cat[2])
    metrics_MAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 5, num_methods_map, cchd_var_map, False,
                                               cchd_MAR_avg_avg[0], cchd_MAR_knn_linreg[0], cchd_MAR_cat_cat[0])
    metrics_MAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 15, num_methods_map, cchd_var_map, False,
                                                cchd_MAR_avg_avg[1], cchd_MAR_knn_linreg[1], cchd_MAR_cat_cat[1])
    metrics_MAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 25, num_methods_map, cchd_var_map, False,
                                                cchd_MAR_avg_avg[2], cchd_MAR_knn_linreg[2], cchd_MAR_cat_cat[2])
    metrics_MNAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 5, num_methods_map, cchd_var_map, False,
                                                cchd_MNAR_avg_avg[0], cchd_MNAR_knn_linreg[0], cchd_MNAR_cat_cat[0])
    metrics_MNAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 15, num_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MNAR_avg_avg[1], cchd_MNAR_knn_linreg[1], cchd_MNAR_cat_cat[1])
    metrics_MNAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 25, num_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MNAR_avg_avg[2], cchd_MNAR_knn_linreg[2], cchd_MNAR_cat_cat[2])

    # Кат.
    metrics_MCAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, cat_methods_map, cchd_var_map, False,
                                                cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
    metrics_MCAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 15, cat_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MCAR_avg_avg[1], cchd_MCAR_knn_linreg[1], cchd_MCAR_cat_cat[1])
    metrics_MCAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 25, cat_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MCAR_avg_avg[2], cchd_MCAR_knn_linreg[2], cchd_MCAR_cat_cat[2])
    metrics_MAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 5, cat_methods_map, cchd_var_map, False,
                                               cchd_MAR_avg_avg[0], cchd_MAR_knn_linreg[0], cchd_MAR_cat_cat[0])
    metrics_MAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 15, cat_methods_map, cchd_var_map, False,
                                                cchd_MAR_avg_avg[1], cchd_MAR_knn_linreg[1], cchd_MAR_cat_cat[1])
    metrics_MAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 25, cat_methods_map, cchd_var_map, False,
                                                cchd_MAR_avg_avg[2], cchd_MAR_knn_linreg[2], cchd_MAR_cat_cat[2])
    metrics_MNAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 5, cat_methods_map, cchd_var_map, False,
                                                cchd_MNAR_avg_avg[0], cchd_MNAR_knn_linreg[0], cchd_MNAR_cat_cat[0])
    metrics_MNAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 15, cat_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MNAR_avg_avg[1], cchd_MNAR_knn_linreg[1], cchd_MNAR_cat_cat[1])
    metrics_MNAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 25, cat_methods_map, cchd_var_map,
                                                 False,
                                                 cchd_MNAR_avg_avg[2], cchd_MNAR_knn_linreg[2], cchd_MNAR_cat_cat[2])

    # 7
    print('MCAR_cat:', (metrics_MCAR5_cat + metrics_MCAR15_cat + metrics_MCAR25_cat) / 3, sep='\n')
    print('MAR_cat:', (metrics_MAR5_cat + metrics_MAR15_cat + metrics_MAR25_cat) / 3, sep='\n')
    print('MNAR_cat:', (metrics_MNAR5_cat + metrics_MNAR15_cat + metrics_MNAR25_cat) / 3, sep='\n')

    print('MCAR_num:', (metrics_MCAR5_num + metrics_MCAR15_num + metrics_MCAR25_num) / 3, sep='\n')
    print('MAR_num:', (metrics_MAR5_num + metrics_MAR15_num + metrics_MAR25_num) / 3, sep='\n')
    print('MNAR_num:', (metrics_MNAR5_num + metrics_MNAR15_num + metrics_MNAR25_num) / 3, sep='\n')


if __name__ == '__main__':
    main()
