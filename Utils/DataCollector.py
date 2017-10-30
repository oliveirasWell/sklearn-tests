import csv
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


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
