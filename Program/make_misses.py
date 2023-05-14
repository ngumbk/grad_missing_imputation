import numpy as np
import pandas as pd


def replace_random_elements(array, n):
    flat_array = array.flatten()  # делаем одномерный массив из двумерного
    indices = np.random.choice(flat_array.size, n, replace=False)  # выбираем случайные индексы
    flat_array[indices] = 1  # заменяем выбранные элементы на 1
    return flat_array.reshape(array.shape)  # возвращаем двумерный массив


def make_mcar(dataset, area='All', miss_percent=5):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    area='All' - аргумент, принимающий наименование одного из столбцов dataset в виде строки. В случае, если не указан, пропуски моделируются во всем датасете
    miss_percent=5 - процент моделируемых пропусков в указанной области в виде числа
    '''
    dataset = dataset.copy()
    if area == 'All':
        elements_number = dataset.shape[0] * dataset.shape[1]
        missing_elements_number = round(miss_percent * elements_number / 100)
        real_miss_percent = missing_elements_number / elements_number * 100  # может не равняться miss_percent, т.к. нужно целое кол-во элементов

        data_missing_indicator = np.zeros(dataset.shape)  # массив индикаторов размером dataset'a
        data_missing_indicator = replace_random_elements(data_missing_indicator, missing_elements_number)

        dataset[data_missing_indicator == 1] = np.nan
    else:
        elements_number = dataset.shape[0]
        missing_elements_number = round(miss_percent * elements_number / 100)
        real_miss_percent = missing_elements_number / elements_number * 100  # может не равняться miss_percent, т.к. нужно целое кол-во элементов

        data_missing_indicator = np.zeros(dataset.shape[0])  # массив индикаторов размером dataset'a
        data_missing_indicator = replace_random_elements(data_missing_indicator, missing_elements_number)

        dataset.loc[data_missing_indicator == 1, area] = np.nan

    print("'" + area + "'", str(miss_percent) + '%', str(round(real_miss_percent, 3)) + '%', sep='; ', end='.\n')

    return dataset


def add_missing_prob(x, area, p_miss):
    if pd.isna(x[area]):
        return x[area]
    else:
        return np.random.choice([x[area], np.nan], p=[1-p_miss[x.name], p_miss[x.name]])


def make_mar(dataset, area, area_dependent, miss_parameter=5, ascending=False, use_prob=False):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    area - аргумент, принимающий наименование одного из столбцов dataset в виде строки, в данном столбце будут созданы пропуски
    area_dependent - аргумент, принимающий наименование одного из столбцов dataset в виде строки, от значений в этом столбце будут зависеть вероятности пропусков в столбце area
    miss_parameter=5 - параметр, применяемый при составлении вектора индикаторов пропусков
    ascending=False - если False, то чем больше значение элемента, тем больше вероятность пропуска
    use_prob=False - если False, то miss_parameter интерпретируется как процент пропусков, которые требуется создать
    '''
    dataset = dataset.copy()

    elements_number = dataset.shape[0]
    missing_elements_number = round(miss_parameter * elements_number / 100)

    if use_prob:
        if ascending:
            p_miss = dataset[area_dependent].apply(
                lambda x: (1 - x / dataset[area_dependent].max()) * miss_parameter / 100)
        else:
            p_miss = dataset[area_dependent].apply(lambda x: x / dataset[area_dependent].max() * miss_parameter / 100)
        dataset[area] = dataset.apply(lambda x: add_missing_prob(x, area, p_miss), axis=1)

    else:
        if ascending:
            # По индексу можно получить макс. значение из n чисел для отсротированного Series.unique()
            max_unique_value_id = len(
                dataset.sort_values(by=area_dependent)[area_dependent].head(missing_elements_number).unique()) - 1
            sorted_series = dataset.sort_values(by=area_dependent).reset_index()[area_dependent]

            miss_n_i = missing_elements_number - len(sorted_series[sorted_series <
                                                                   dataset.sort_values(by=area_dependent)[
                                                                       area_dependent].unique()[max_unique_value_id]])
            miss_b_i = missing_elements_number - miss_n_i

            a = miss_b_i
            b = a + len(sorted_series[sorted_series == dataset.sort_values(by=area_dependent)[area_dependent].unique()[
                max_unique_value_id]])

            mask_shuffled = ([True] * miss_n_i + [False] * (b - miss_b_i - miss_n_i))
            np.random.shuffle(mask_shuffled)
            mask_shuffled = [True] * a + mask_shuffled + [False] * (
                        elements_number - b)  # где True - там делать пропуск
            dataset = dataset.sort_values(by=area_dependent)
        else:
            # По индексу можно получить макс. значение из n чисел для отсротированного Series.unique()
            max_unique_value_id = len(dataset.sort_values(by=area_dependent, ascending=False)[area_dependent].head(
                missing_elements_number).unique()) - 1
            sorted_series = dataset.sort_values(by=area_dependent, ascending=False).reset_index()[area_dependent]

            miss_n_i = missing_elements_number - len(sorted_series[sorted_series >
                                                                   dataset.sort_values(by=area_dependent,
                                                                                       ascending=False)[
                                                                       area_dependent].unique()[max_unique_value_id]])
            miss_b_i = missing_elements_number - miss_n_i

            a = miss_b_i
            b = a + len(sorted_series[sorted_series ==
                                      dataset.sort_values(by=area_dependent, ascending=False)[area_dependent].unique()[
                                          max_unique_value_id]])

            mask_shuffled = ([True] * miss_n_i + [False] * (b - miss_b_i - miss_n_i))
            np.random.shuffle(mask_shuffled)
            mask_shuffled = [True] * a + mask_shuffled + [False] * (
                        elements_number - b)  # где True - там делать пропуск
            dataset = dataset.sort_values(by=area_dependent, ascending=False)

        dataset.loc[mask_shuffled, area] = np.nan
        dataset = dataset.sort_index()

    real_miss_percent = len(dataset[dataset[area].isna()]) / elements_number * 100
    print("'" + area + "'", "'" + area_dependent + "'", miss_parameter, str(round(real_miss_percent, 3)) + '%',
          sep='; ', end='.\n')

    return dataset


def make_mnar(dataset, area, miss_parameter=5, ascending=False, use_prob=False):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    area - аргумент, принимающий наименование одного из столбцов dataset в виде строки
    miss_parameter=5 - параметр, применяемый при составлении вектора индикаторов пропусков
    ascending=False - если False, то чем больше значение элемента, тем больше вероятность пропуска
    use_prob=False - если False, то miss_parameter интерпретируется как процент пропусков, которые требуется создать
    '''
    dataset = dataset.copy()

    elements_number = dataset.shape[0]
    missing_elements_number = round(miss_parameter * elements_number / 100)

    if use_prob:
        if ascending:
            p_miss = dataset[area].apply(lambda x: (1 - x / dataset[area].max()) * miss_parameter / 100)
        else:
            p_miss = dataset[area].apply(lambda x: x / dataset[area].max() * miss_parameter / 100)
        dataset[area] = dataset.apply(lambda x: add_missing_prob(x, area, p_miss), axis=1)
    else:
        if ascending:
            # По индексу можно получить макс. значение из n чисел для отсротированного Series.unique()
            max_unique_value_id = len(dataset.sort_values(by=area)[area].head(missing_elements_number).unique()) - 1
            sorted_series = dataset.sort_values(by=area).reset_index()[area]

            miss_n_i = missing_elements_number - len(
                sorted_series[sorted_series < dataset.sort_values(by=area)[area].unique()[max_unique_value_id]])
            miss_b_i = missing_elements_number - miss_n_i

            a = miss_b_i
            b = a + len(
                sorted_series[sorted_series == dataset.sort_values(by=area)[area].unique()[max_unique_value_id]])

            mask_shuffled = ([True] * miss_n_i + [False] * (b - miss_b_i - miss_n_i))
            np.random.shuffle(mask_shuffled)
            mask_shuffled = [True] * a + mask_shuffled + [False] * (
                        elements_number - b)  # где True - там делать пропуск
            dataset = dataset.sort_values(by=area)
        else:
            # По индексу можно получить макс. значение из n чисел для отсротированного Series.unique()
            max_unique_value_id = len(
                dataset.sort_values(by=area, ascending=False)[area].head(missing_elements_number).unique()) - 1
            sorted_series = dataset.sort_values(by=area, ascending=False).reset_index()[area]

            miss_n_i = missing_elements_number - len(sorted_series[sorted_series >
                                                                   dataset.sort_values(by=area, ascending=False)[
                                                                       area].unique()[max_unique_value_id]])
            miss_b_i = missing_elements_number - miss_n_i

            a = miss_b_i
            b = a + len(sorted_series[sorted_series == dataset.sort_values(by=area, ascending=False)[area].unique()[
                max_unique_value_id]])

            mask_shuffled = ([True] * miss_n_i + [False] * (b - miss_b_i - miss_n_i))
            np.random.shuffle(mask_shuffled)
            mask_shuffled = [True] * a + mask_shuffled + [False] * (
                        elements_number - b)  # где True - там делать пропуск

            dataset = dataset.sort_values(by=area, ascending=False)

    dataset.loc[mask_shuffled, area] = np.nan
    dataset = dataset.sort_index()

    real_miss_percent = len(dataset[dataset[area].isna()]) / elements_number * 100
    print("'" + area + "'", miss_parameter, str(round(real_miss_percent, 3)) + '%', sep='; ', end='.\n')
    return dataset


def make_set_of_datasets(dataset, mechanism, name='', miss_parameters=[5, 15, 25]):
    '''
    dataset - исходный набор данных в виде pandas dataframe
    mechanism - механизм пропусков, допускаемых в данном наборе датасетов. Может принимать значение: MCAR, MAR или MNAR
    name='' - название ключей словаря с набором датасетов
    miss_parameters=[5, 15, 25] - список параметров пропуска, принимает 3 параметра, означающих примерные доли пропусков в результирующих наборах
    '''
    dataset = dataset.copy()
    c = dataset.shape[1]  # количество переменных в датасете
    df_sets = []  # список словарей

    for i in range(3):  # создаем 3 набора датасетов, соответственно заданным параметрам пропусков
        df_set = {}  # словарь для c датасетов
        keys_set = [f'{name}_{mechanism}{miss_parameters[i]}_{dataset.columns[j]}' for j in
                    range(0, c)]  # создаем список ключей для получаемых датафреймов
        for j in range(c):  # создаем c датафреймов с пропусками
            if mechanism == 'MCAR':
                df_set[keys_set[j]] = make_mcar(dataset, area=dataset.columns[j], miss_percent=miss_parameters[i])
            elif mechanism == 'MAR':
                df_set[keys_set[j]] = make_mar(dataset, area=dataset.columns[j],
                                               area_dependent=dataset.columns[(j - 1) % c],
                                               miss_parameter=miss_parameters[i])
            elif mechanism == 'MNAR':
                df_set[keys_set[j]] = make_mnar(dataset, area=dataset.columns[j], miss_parameter=miss_parameters[i])
            else:
                print('Ошибка в названии механизма пропуска')
                return None
        df_sets.append(df_set)  # добавляем словарь в список
    return df_sets
