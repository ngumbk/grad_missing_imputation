import numpy as np
import pandas as pd
from model_misses import *
from make_set import make_set_of_datasets
from imputation_functions import *
from make_metrics_functions import *
import streamlit as st


def main():
    cchd_data = pd.read_csv('../../ВКР/Datasets/processed_cleveland.csv')
    cchd_data.head(6)
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

    # Создадим наборы с пропусками
    cchd_data_set_MCAR = make_set_of_datasets(cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_data_set_MAR = make_set_of_datasets(cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25])
    cchd_data_set_MNAR = make_set_of_datasets(cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25])

    # Восстановим пропуски в наборах с помощью разных методов
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

    # Подсчет метрик для восстановленных численных переменных
    num_methods_map = {0: 'Median Imputation', 1: 'Linear Regression', 2: 'Boosted Decision Tree'}

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

    # Подсчет метрик для восстановленных категориальных переменных
    cat_methods_map = {0: 'Mode Imputation', 1: 'K-Nearest Neighbours', 2: 'Boosted Decision Tree'}

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


if __name__ == '__main__':
    main()
