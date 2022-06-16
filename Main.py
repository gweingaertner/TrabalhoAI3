import random as rd
import numpy as np
import math as mat

roleta = []

lista_items = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
count = 1
for i in lista_items:
    for j in range(count):
        roleta.append(i)
    count += 1
data = np.loadtxt('cidades.mat')
print(data[0])

