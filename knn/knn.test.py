import csv
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


#################################################################################################################################
class DataCollector:
    def __init__(self):
        print("Started")

    def lerCsv(self, nome_ficheiro):
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


#################################################################################################################################
class SearchAnalyzer:
    def __init__(self, dataset, param_grid_in, model_selection):
        self.dataset = dataset

        data = DataCollector()
        self.X, self.y = data.lerCsv(dataset)

        self.search_methods = [('Grid', GridSearchCV(model_selection, param_grid_in, cv=10, scoring='accuracy')),
                               ('Random', RandomizedSearchCV(model_selection, param_grid_in, cv=10, scoring='accuracy', n_iter=30, random_state=5))]

    def getMelhoresResultados(self):
        for name, search_method in self.search_methods:
            print('----------------------------------------------------')
            print('Metodo de busca: ' + name)
            search_method.fit(self.X, self.y)

            print("Melhor resultado:")
            print(search_method.best_score_)
            print()

            print("Melhor parametros:")
            print(search_method.best_params_)
            print()

            self.printResultados(search_method)

    def printResultados(self, search_method):
        print("Lista de scores:")
        means = search_method.cv_results_['mean_test_score']
        stds = search_method.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, search_method.cv_results_['params']):
            print("%0.6f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print('----------------------------------------------------')
        print()


#################################################################################################################################
def main():
    data_sets = ['datasets/iris.data', 'datasets/ionosphere.data']
    param_grid_in = dict(n_neighbors=list(range(1, 31)), weights=['uniform', 'distance'])
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance')

    for dataset in data_sets:
        print('----------------------------------------------------')
        print('Dataset:' + dataset)
        SearchAnalyzer(dataset, param_grid_in, knn).getMelhoresResultados()
        print('----------------------------------------------------')


#################################################################################################################################
if __name__ == "__main__":
    main()
