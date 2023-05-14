from imputation_functions import *
import streamlit as st


def impute_set(df_sets, var_map, cat_impute_function, num_impute_function, dataset, mechanism, name,
               miss_parameters=None, avg_impute_mode='median'):
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

    if miss_parameters is None:
        miss_parameters = [5, 15, 25]
    imputed_df_sets = []  # список наборов, который будет возвращен
    c = len(df_sets[0])  # количество переменных в каждом датасете

    output_map = {impute_average: 'Восстановление средним',
                  impute_linreg: 'Линейная регрессия',
                  impute_knn: 'k-ближайших соседей',
                  impute_catboost_cat: 'Решающее дерево (кат.)',
                  impute_catboost_num: 'Решающее дерево (числ.)'}
    st.write(f'Восстановление набора с пропусками {mechanism} в объеме {miss_parameters}')
    st.write(f'Метод восстановления категориальных переменных: {output_map[cat_impute_function]},')
    st.write(f'Метод восстановления численных переменных: {output_map[num_impute_function]}')

    progress_bar = st.progress(0)

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
            progress_bar.progress(int(100 * (j + i * c) / c / 3))
        imputed_df_sets.append(imputed_df_set)
    progress_bar.progress(100)

    return imputed_df_sets
