import csv

import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


class DataCollector:
    def __init__(self):
        print("Started")

    def getEntrada(self, nome_ficheiro):
        vetor_entrada = []

        with open(nome_ficheiro, 'r') as ficheiro:
            reader = csv.reader(ficheiro)
            try:
                for linha in reader:
                    novaLinha = []
                    for coluna in linha[:-1]:
                        novaLinha.append(float(coluna))
                    novaLinha.append(linha[-1])
                    vetor_entrada.append(novaLinha)
            except csv.Error as e:
                sys.exit('ficheiro %s, linha %d: %s' % (nome_ficheiro, reader.line_num, e))
        return self.getVetorXySeparado(vetor_entrada)

    def getVetorXySeparado(self, vetor_entrada):
        x = []
        y = []

        for linha in vetor_entrada:
            linha_x_aux = []
            for coluna in linha[:-1]:
                clouna_temp = coluna
                linha_x_aux.append(clouna_temp)
            x.append(linha_x_aux)
            y.append(linha[-1])

        return x, y


def main():
    data = DataCollector()
    data_sets = ['iris', 'ionosphere']

    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']
    param_grid_in = dict(n_neighbors=k_range, weights=weight_options)

    print('----------------------------------------------------')
    for dataset in data_sets:

        print('Dataset:' + dataset)

        X, y = data.getEntrada('datasets/' + dataset + '.data')

        knn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='ball_tree')

        search_methods = [('Grid', GridSearchCV(knn, param_grid_in, cv=10, scoring='accuracy')),
                          ('Random',
                           RandomizedSearchCV(knn, param_grid_in, cv=10, scoring='accuracy', n_iter=30, random_state=5))]

        for name, search_method in search_methods:
            print('----------------------------------------------------')
            print('Search method: ' + name)
            search_method.fit(X, y)
            print("Best score:")
            print(search_method.best_score_)
            print()

            print("Best params:")
            print(search_method.best_params_)
            print()

            # print_scores(search_method)


def print_scores(search_method):
    print("Lista de scores:")
    means = search_method.cv_results_['mean_test_score']
    stds = search_method.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search_method.cv_results_['params']):
        print("%0.6f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('----------------------------------------------------')
    print()


if __name__ == "__main__":
    main()
