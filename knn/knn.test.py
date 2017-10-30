import csv

import sys
from sklearn.neighbors import KNeighborsClassifier


class DataCollector:
    def __init__(self):
        print("Started")

    def getEntrada(self, nome_ficheiro):
        vetorEntrada = []

        with open(nome_ficheiro, 'r') as ficheiro:
            reader = csv.reader(ficheiro)
            try:
                for linha in reader:
                    novaLinha = []
                    for coluna in linha[:-1]:
                        novaLinha.append(float(coluna))
                    novaLinha.append(linha[-1])
                    vetorEntrada.append(novaLinha)
            except csv.Error as e:
                sys.exit('ficheiro %s, linha %d: %s' % (nome_ficheiro, reader.line_num, e))
        return self.getVetorXySeparado(vetorEntrada)

    def getVetorXySeparado(self, vetorEntrada):
        x = []
        y = []

        for linha in vetorEntrada:
            linhaXAux = []
            for coluna in linha[:-1]:
                clounaTemp = coluna
                linhaXAux.append(clounaTemp)
            x.append(linhaXAux)
            y.append(linha[-1])
        return x, y


if __name__ == "__main__":
    data = DataCollector()
    X, y = data.getEntrada('datasets/iris.data')

    neigh = KNeighborsClassifier(n_neighbors=3)

    neigh.fit(X, y)

    print(neigh.predict([[5.1, 3.5, 1.4, 0.2]]))

#    print(neigh.predict_proba(['Iris-setosa']))
