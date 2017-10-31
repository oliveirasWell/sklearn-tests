from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from Utils.DataCollector import DataCollector


class SearchAnalyzer:
    def __init__(self, dataset, param_grid_in, model_selection, allow_random=True):
        self.dataset = dataset
        self.X, self.y = DataCollector().lerCsv(dataset)
        self.allow_random = allow_random
        self.search_methods = [('Grid', GridSearchCV(model_selection, param_grid_in, cv=10, scoring='accuracy', n_jobs=-1)),
                               ('Random', RandomizedSearchCV(model_selection, param_grid_in, cv=10, scoring='accuracy', n_iter=30, random_state=5))]

    def getMelhoresResultados(self):
        for name, search_method in self.search_methods:

            if name == 'Random' and not self.allow_random:
                return

            print('----------------------------------------------------')
            print('Metodo de busca: ' + name)
            search_method.fit(self.X, self.y)

            print("Melhor resultado:")
            print(search_method.best_score_)
            print()

            print("Melhor parametros:")
            print(search_method.best_params_)
            print()

            self.printListaResultados(search_method)

    def printListaResultados(self, search_method):
        print("Lista de scores:")
        means = search_method.cv_results_['mean_test_score']
        stds = search_method.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, search_method.cv_results_['params']):
            print("%0.6f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print('----------------------------------------------------')
        print()
