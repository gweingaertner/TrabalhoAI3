import random as rd
import numpy as np
import math as mat

data = np.loadtxt('C:/Users/lksmi/Desktop/TrabalhoAlgoritmoGenetico/cidades.mat')
x = data[0]
y = data[1]
cidades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
roleta = [9,
          8,8,
          7,7,7,
          6,6,6,6,
          5,5,5,5,5,
          4,4,4,4,4,4,
          3,3,3,3,3,3,3,
          2,2,2,2,2,2,2,2,
          1,1,1,1,1,1,1,1,1,
          0,0,0,0,0,0,0,0,0,0]

cromossomo = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

populacao = 20


def gerarPopulacao(qnt):
    matrix = []
    for i in range(qnt):
        linha = []
        cidadestemp = cidades.copy()
        for x in range(qnt):
            cidade = rd.choice(cidadestemp)
            linha.append(cidade)
            cidadestemp.remove(cidade)
        matrix.append(linha)

    return matrix

def calcularAptidao(matrixDist):
    matrixAptidao = []

    for linha in matrixDist:
        matrixAptidao.append(np.sum(linha))
    return matrixAptidao

def main():

    matrix = gerarPopulacao(populacao)

    matrixDist = calcularMatrixDistancia(matrix)
    matrixAptidao = calcularAptidao(matrixDist)
    matrixAptidao, matrixDist, matrix = ordenar(matrixAptidao, matrixDist, matrix)
    removerMenosAptos(matrixAptidao, matrixDist, matrix)


    for interacoes in range(1000):
        print(interacoes)
        descendentes = gerarDescentes(matrix)
        matrixDistFilho = calcularMatrixDistancia(descendentes)
        matrixAptidaoFilho = calcularAptidao(matrixDistFilho)
        matrixAptidaoFilho, matrixDistFilho, descendentes = ordenar(matrixAptidaoFilho, matrixDistFilho, descendentes)
        removerMenosAptos(matrixAptidaoFilho, matrixDistFilho, descendentes)

        for membro in descendentes:
            matrix.append(membro)

        matrixDist = calcularMatrixDistancia(matrix)
        matrixAptidao = calcularAptidao(matrixDist)
        matrixAptidao, matrixDist, matrix = ordenar(matrixAptidao, matrixDist, matrix)
        removerMenosAptos(matrixAptidao, matrixDist, matrix)
        descendentes.clear()
        matrixDistFilho.clear()
        matrixAptidaoFilho.clear()

    for linha in matrixAptidao:
        print(linha)
    for linha in matrix:
        print(linha)


def gerarDescentes(matrix):
    descendentes = []
    for idxp in range(10):
        pai1, pai2 = sortearValores()
        filho1, filho2 = crossOver(matrix[pai1], matrix[pai2])
        mutacao(filho1)
        mutacao(filho2)
        descendentes.append(filho1)
        descendentes.append(filho2)
    return descendentes


def removerMenosAptos(matrixAptidao,matrixDist,matrix):
    for x in range(10):
        matrixAptidao.pop()
        matrixDist.pop()
        matrix.pop()

    return matrixAptidao,matrixDist,matrix

def crossOver(pai1,pai2):
    local = rd.choice(cromossomo)
    filho1 = pai1.copy()
    filho2 = pai2.copy()

    aux = filho1[local]
    filho1[local]= filho2[local]
    hv,idx = hasValue(filho2,aux)
    filho2[local]= aux
    local = idx

    while(hv):

        aux = filho1[local]
        filho1[local]= filho2[local]
        hv,idx = hasValue(filho2,aux)
        if(idx == local):
            hv = False
        filho2[local]= aux
        local = idx

    return filho1,filho2

def mutacao(membro):
    cromossomoaux = cromossomo.copy()
    op1 = rd.choice(cromossomoaux)
    cromossomoaux.remove(op1)
    op2 = rd.choice(cromossomoaux)

    aux = membro[op1]
    membro[op1] = membro[op2]
    membro[op2] = aux
    return membro

def hasValue(matrix,value):
    for idx,valores in enumerate(matrix, start=0):
        if(valores == value):
            return True,idx
    return False,-1



def ordenar(matrixAptidao,matrixDistancia,matrix):

    idx = 0
    while(idx < 19):
        if(matrixAptidao[idx] > matrixAptidao[idx+1]):
            aux = matrixAptidao[idx]
            matrixAptidao[idx] = matrixAptidao[idx+1]
            matrixAptidao[idx+1] = aux
            aux = matrixDistancia[idx]
            matrixDistancia[idx] = matrixDistancia[idx+1]
            matrixDistancia[idx+1] = aux
            aux = matrix[idx]
            matrix[idx] = matrix[idx+1]
            matrix[idx+1] = aux
            idx = 0
        else:
            idx = idx+1
    return matrixAptidao,matrixDistancia,matrix

def sortearValores():
    pai2 = 0
    pai1 = 0

    while (pai2 == pai1):
        pai1 = rd.choice(roleta)
        pai2 = rd.choice(roleta)
    return pai1,pai2

def calcularMatrixDistancia(matrix):
    matrixDistancia = []
    for linha in matrix:
        linha.append(linha[0])
        line = []
        for ind in range(20):
            xtemp = x[linha[ind]-1]
            ytemp = y[linha[ind+1]-1]
            line.append(mat.sqrt(xtemp**2 + ytemp**2))
        matrixDistancia.append(line)
        linha.pop()
    return matrixDistancia

if __name__ == '__main__':
    main()
