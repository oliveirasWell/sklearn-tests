from sklearn.neighbors import KNeighborsClassifier

from Utils.SearchAnalyzer import SearchAnalyzer


def main():
    data_sets = ['datasets/iris.data', 'datasets/ionosphere.data']
    param_grid_in = dict(n_neighbors=list(range(1, 31)), weights=['uniform', 'distance'])
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance')

    for dataset in data_sets:
        print('----------------------------------------------------')
        print('Dataset:' + dataset)
        SearchAnalyzer(dataset, param_grid_in, knn).getMelhoresResultados()
        print('----------------------------------------------------')


if __name__ == "__main__":
    main()
