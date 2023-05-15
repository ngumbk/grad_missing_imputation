import matplotlib.pyplot as plt
import seaborn as sns
from make_misses import *
from impute_set_function import *
from make_metrics_functions import *
from Util_functions import *


def main():
    # Программа работает ТОЛЬКО с датасетом CCHD
    # 1. Чтение и вывод датасета
    st.markdown('# Демонстрация работы программы для датасета *Cleveland Clinic Heart Disease Dataset*')

    cchd_data = pd.read_csv('../../ВКР/Datasets/processed_cleveland.csv')
    cchd_data.index.name = 'id'

    cchd_data.loc[cchd_data['ca'] == '?', 'ca'] = cchd_data['ca'].mode().iloc[0]
    cchd_data.loc[cchd_data['thal'] == '?', 'thal'] = cchd_data['thal'].mode().iloc[0]
    cchd_data = cchd_data.astype({'ca': 'int64', 'thal': 'int64'})

    st.markdown('## Описание переменных датасета')
    with st.expander('Описание переменных'):
        st.markdown(read_markdown_file("markdowns/var_description_table.md"))
        st.markdown('### Таблица корреляций')
        st.write('В таблице приведены коэффициенты корреляций, посчитанные для всех пар переменных.')
        fig, ax = plt.subplots()
        sns.heatmap(cchd_data.corr(), ax=ax, annot=True, annot_kws={'fontsize': 5})
        st.write(fig)

    st.markdown('## Чтение и обзор датасета')
    with st.expander('Часть объектов датасета'):
        st.write('Выведем первые 10 объектов датасета:')
        st.dataframe(cchd_data.head(10))

    # Изменения в датасете
    st.markdown('## Описание изменений в датасете')
    st.markdown(read_markdown_file("markdowns/dataset_changes_description.md"))
    with st.expander('Таблица с описанием изменений'):
        st.markdown(read_markdown_file("markdowns/dataset_var_changes_table.md"))

    st.markdown('## Измененный датасет')

    cchd_data['cp'] = cchd_data['cp'].replace({2: 1, 3: 2, 4: 3})
    cchd_data['restecg'] = cchd_data['restecg'].replace({2: 1})
    cchd_data = cchd_data.drop(columns=['fbs'])
    cchd_data['slope'] = cchd_data['slope'].replace({1: 0, 2: 1, 3: 1})
    cchd_data['ca'] = cchd_data['ca'].replace({3: 2})
    cchd_data['thal'] = cchd_data['thal'].replace({7: 1, 6: 1, 3: 0})
    cchd_data['num'] = cchd_data['num'].replace({4: 3})

    with st.expander('Измененный датасет'):
        st.write('Первые 10 объектов измененного датасета:')
        st.dataframe(cchd_data.head(10))

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

    # 2. Демонстрация появления пропусков
    st.markdown('## Датасеты с пропусками')
    st.markdown('Приведенные ниже датасеты демонстрируют работу функций создания пропусков с учетом разных механизмов '
                'появления пропусков.')

    with st.expander('Примеры датасетов с разными механизмами пропусков'):
        tab_mcar, tab_mar, tab_mnar = st.tabs(["MCAR", "MAR", "MNAR"])
        with tab_mcar:
            st.markdown('### MCAR 25% в переменной "slope":')
            st.dataframe(make_mcar(cchd_data, area='slope', miss_percent=25))
        with tab_mar:
            st.markdown('### MAR 30% в переменной "sex". Наличие пропусков зависит от значений в переменной "oldpeak" '
                        'по убыванию:')
            st.dataframe(make_mar(cchd_data, area='sex', area_dependent='oldpeak', miss_parameter=30, ascending=False))
        with tab_mnar:
            st.markdown('### MNAR 50% в переменной "oldpeak". Наличие пропусков зависит от значений в самой '
                        'переменной по возрастанию:')
            st.dataframe(make_mnar(cchd_data, area='oldpeak', miss_parameter=49, ascending=True))

    # 4. Создание наборов с пропусками
    st.markdown('## Восстановление наборов пропусков')
    st.write('После выполнения восстановления будут отображены таблицы с метриками.')
    st.write('Операция восстановления набора достаточно ресурсоемкая. Не стоит запускать её, если вы экономите заряд '
             'ноутбука или у вас очень слабый процессор.')
    st.write('Время выполнения: ~5 минут.')

    if st.button('Создать наборы (9 шт.) таблиц с пропусками и начать их восстановление'):
        cchd_data_set_MCAR = make_set_of_datasets(cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
        cchd_data_set_MAR = make_set_of_datasets(cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25])
        cchd_data_set_MNAR = make_set_of_datasets(cchd_data, 'MNAR', 'cchd_data', miss_parameters=[5, 15, 25])

        # 5. Восстановление наборов с пропусками
        cchd_MCAR_avg_avg = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_average, impute_average,
                                       cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25],
                                       avg_impute_mode='median')
        cchd_MCAR_knn_linreg = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_knn, impute_linreg,
                                          cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
        cchd_MCAR_cat_cat = impute_set(cchd_data_set_MCAR, cchd_var_map, impute_catboost_cat, impute_catboost_num,
                                       cchd_data, 'MCAR', 'cchd_data', miss_parameters=[5, 15, 25])
        cchd_MAR_avg_avg = impute_set(cchd_data_set_MAR, cchd_var_map, impute_average, impute_average,
                                      cchd_data, 'MAR', 'cchd_data', miss_parameters=[5, 15, 25],
                                      avg_impute_mode='median')
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

        # 6. Создание матриц метрик
        st.markdown('## Расчет метрик')
        st.write('Для сравнения качества восстановления, выбраны F1-метрика и RMSE-метрика для категориальных и '
                 'численных переменных, соответственно.')
        num_methods_map = {0: 'Median Imputation', 1: 'Linear Regression', 2: 'Boosted Decision Tree'}
        cat_methods_map = {0: 'Mode Imputation', 1: 'K-Nearest Neighbours', 2: 'Boosted Decision Tree'}

        # Численные метрики (RMSE)
        metrics_MCAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, num_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
        metrics_MCAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 15, num_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MCAR_avg_avg[1], cchd_MCAR_knn_linreg[1],
                                                     cchd_MCAR_cat_cat[1])
        metrics_MCAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 25, num_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MCAR_avg_avg[2], cchd_MCAR_knn_linreg[2],
                                                     cchd_MCAR_cat_cat[2])
        metrics_MAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 5, num_methods_map, cchd_var_map,
                                                   False,
                                                   cchd_MAR_avg_avg[0], cchd_MAR_knn_linreg[0], cchd_MAR_cat_cat[0])
        metrics_MAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 15, num_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MAR_avg_avg[1], cchd_MAR_knn_linreg[1], cchd_MAR_cat_cat[1])
        metrics_MAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 25, num_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MAR_avg_avg[2], cchd_MAR_knn_linreg[2], cchd_MAR_cat_cat[2])
        metrics_MNAR5_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 5, num_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MNAR_avg_avg[0], cchd_MNAR_knn_linreg[0], cchd_MNAR_cat_cat[0])
        metrics_MNAR15_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 15, num_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MNAR_avg_avg[1], cchd_MNAR_knn_linreg[1],
                                                     cchd_MNAR_cat_cat[1])
        metrics_MNAR25_num = make_num_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 25, num_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MNAR_avg_avg[2], cchd_MNAR_knn_linreg[2],
                                                     cchd_MNAR_cat_cat[2])

        # Категориальные метрики (F1)
        metrics_MCAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 5, cat_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MCAR_avg_avg[0], cchd_MCAR_knn_linreg[0], cchd_MCAR_cat_cat[0])
        metrics_MCAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 15, cat_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MCAR_avg_avg[1], cchd_MCAR_knn_linreg[1],
                                                     cchd_MCAR_cat_cat[1])
        metrics_MCAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MCAR', 25, cat_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MCAR_avg_avg[2], cchd_MCAR_knn_linreg[2],
                                                     cchd_MCAR_cat_cat[2])
        metrics_MAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 5, cat_methods_map, cchd_var_map,
                                                   False,
                                                   cchd_MAR_avg_avg[0], cchd_MAR_knn_linreg[0], cchd_MAR_cat_cat[0])
        metrics_MAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 15, cat_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MAR_avg_avg[1], cchd_MAR_knn_linreg[1], cchd_MAR_cat_cat[1])
        metrics_MAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MAR', 25, cat_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MAR_avg_avg[2], cchd_MAR_knn_linreg[2], cchd_MAR_cat_cat[2])
        metrics_MNAR5_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 5, cat_methods_map, cchd_var_map,
                                                    False,
                                                    cchd_MNAR_avg_avg[0], cchd_MNAR_knn_linreg[0], cchd_MNAR_cat_cat[0])
        metrics_MNAR15_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 15, cat_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MNAR_avg_avg[1], cchd_MNAR_knn_linreg[1],
                                                     cchd_MNAR_cat_cat[1])
        metrics_MNAR25_cat = make_cat_metrics_matrix(cchd_data, 'cchd_data', 'MNAR', 25, cat_methods_map, cchd_var_map,
                                                     False,
                                                     cchd_MNAR_avg_avg[2], cchd_MNAR_knn_linreg[2],
                                                     cchd_MNAR_cat_cat[2])

        # Вывод матриц метрик, усредненных по разным количествам пропусков
        with st.expander('Матрицы метрик'):
            st.markdown('### F1-метрика восстановления кат. MCAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MCAR5_cat + metrics_MCAR15_cat + metrics_MCAR25_cat) / 3, ax=ax, annot=True)
            st.write(fig)

            st.markdown('### F1-метрика восстановления кат. MAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MAR5_cat + metrics_MAR15_cat + metrics_MAR25_cat) / 3, ax=ax, annot=True)
            st.write(fig)

            st.markdown('### F1-метрика восстановления кат. MNAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MNAR5_cat + metrics_MNAR15_cat + metrics_MNAR25_cat) / 3, ax=ax, annot=True)
            st.write(fig)

            st.markdown('### RMSE-метрика восстановления числ. MCAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MCAR5_num + metrics_MCAR15_num + metrics_MCAR25_num) / 3, ax=ax, annot=True)
            st.write(fig)

            st.markdown('### RMSE-метрика восстановления числ. MAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MAR5_num + metrics_MAR15_num + metrics_MAR25_num) / 3, ax=ax, annot=True)
            st.write(fig)

            st.markdown('### RMSE-метрика восстановления числ. MNAR пропусков')
            fig, ax = plt.subplots()
            sns.heatmap((metrics_MNAR5_num + metrics_MNAR15_num + metrics_MNAR25_num) / 3, ax=ax, annot=True)
            st.write(fig)

        with st.expander('Графики метрик'):
            st.markdown('### Численные переменные')
            # MCAR num
            list_of_MCAR_num = list(zip(metrics_MCAR5_num.sum(axis=1) / 5, metrics_MCAR15_num.sum(axis=1) / 5,
                                        metrics_MCAR25_num.sum(axis=1) / 5))
            MCAR_num_stat = pd.DataFrame(np.array(list_of_MCAR_num).T, columns=[num_methods_map[i] for i in range(3)],
                                         index=[5, 15, 25])
            fig, ax = plt.subplots()
            MCAR_num_graph = sns.lineplot(data=MCAR_num_stat, ax=ax)
            MCAR_num_graph.set_xticks(range(5, 26, 10))
            MCAR_num_graph.set_yticks(range(3, 20))
            plt.title("RMSE для MCAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("RMSE-метрика")

            st.write(fig)

            # MAR num
            list_of_MAR_num = list(zip(metrics_MAR5_num.sum(axis=1) / 5, metrics_MAR15_num.sum(axis=1) / 5,
                                       metrics_MAR25_num.sum(axis=1) / 5))
            MAR_num_stat = pd.DataFrame(np.array(list_of_MAR_num).T, columns=[num_methods_map[i] for i in range(3)],
                                        index=[5, 15, 25])
            fig, ax = plt.subplots()
            MAR_num_graph = sns.lineplot(data=MAR_num_stat, ax=ax)
            MAR_num_graph.set_xticks(range(5, 26, 10))
            MAR_num_graph.set_yticks(range(3, 20))
            plt.title("RMSE для MAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("RMSE-метрика")
            st.write(fig)

            # MNAR num
            list_of_MNAR_num = list(zip(metrics_MNAR5_num.sum(axis=1) / 5, metrics_MNAR15_num.sum(axis=1) / 5,
                                        metrics_MNAR25_num.sum(axis=1) / 5))
            MNAR_num_stat = pd.DataFrame(np.array(list_of_MNAR_num).T, columns=[num_methods_map[i] for i in range(3)],
                                         index=[5, 15, 25])
            fig, ax = plt.subplots()
            MNAR_num_graph = sns.lineplot(data=MNAR_num_stat, ax=ax)
            MNAR_num_graph.set_xticks(range(5, 26, 10))
            MNAR_num_graph.set_yticks(range(3, 20))
            plt.title("RMSE для MNAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("RMSE-метрика")
            st.write(fig)

            st.markdown('### Категориальные переменные')

            # MCAR cat
            list_of_MCAR_cat = list(zip(metrics_MCAR5_cat.sum(axis=1) / 8, metrics_MCAR15_cat.sum(axis=1) / 8,
                                        metrics_MCAR25_cat.sum(axis=1) / 8))
            MCAR_cat_stat = pd.DataFrame(np.array(list_of_MCAR_cat).T, columns=[cat_methods_map[i] for i in range(3)],
                                         index=[5, 15, 25])
            fig, ax = plt.subplots()
            MCAR_cat_graph = sns.lineplot(data=MCAR_cat_stat, ax=ax)
            MCAR_cat_graph.set_xticks(range(5, 26, 10))
            MCAR_cat_graph.set_yticks([0.025 * i for i in range(31, 40)])  # 0.775 - 1
            plt.title("F1 для MCAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("F1-метрика")
            st.write(fig)

            # MAR cat
            list_of_MAR_cat = list(zip(metrics_MAR5_cat.sum(axis=1) / 8, metrics_MAR15_cat.sum(axis=1) / 8,
                                       metrics_MAR25_cat.sum(axis=1) / 8))
            MAR_cat_stat = pd.DataFrame(np.array(list_of_MAR_cat).T, columns=[cat_methods_map[i] for i in range(3)],
                                        index=[5, 15, 25])
            fig, ax = plt.subplots()
            MAR_cat_graph = sns.lineplot(data=MAR_cat_stat, ax=ax)
            MAR_cat_graph.set_xticks(range(5, 26, 10))
            MAR_cat_graph.set_yticks([0.025 * i for i in range(31, 40)])  # 0.775 - 1
            plt.title("F1 для MAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("F1-метрика")
            st.write(fig)

            # MNAR cat
            list_of_MNAR_cat = list(zip(metrics_MNAR5_cat.sum(axis=1) / 8, metrics_MNAR15_cat.sum(axis=1) / 8,
                                        metrics_MNAR25_cat.sum(axis=1) / 8))
            MNAR_cat_stat = pd.DataFrame(np.array(list_of_MNAR_cat).T, columns=[cat_methods_map[i] for i in range(3)],
                                         index=[5, 15, 25])
            fig, ax = plt.subplots()
            MNAR_cat_graph = sns.lineplot(data=MNAR_cat_stat, ax=ax)
            MNAR_cat_graph.set_xticks(range(5, 26, 10))
            MNAR_cat_graph.set_yticks([0.025 * i for i in range(31, 40)])  # 0.775 - 1
            plt.title("F1 для MNAR-пропусков")
            plt.xlabel("Доля пропусков")
            plt.ylabel("F1-метрика")
            st.write(fig)


if __name__ == '__main__':
    main()
