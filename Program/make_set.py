from model_misses import *


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
