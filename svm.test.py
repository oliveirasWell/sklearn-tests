from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from Utils.SearchAnalyzer import SearchAnalyzer


def main():
    data_sets = ['datasets/iris.data', 'datasets/ionosphere.data']

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    knn = SVC()

    for dataset in data_sets:
        print('----------------------------------------------------')
        print('Dataset:' + dataset)
        SearchAnalyzer(dataset, param_grid, knn, allow_random=False).getMelhoresResultados()
        print('----------------------------------------------------')


if __name__ == "__main__":
    main()
