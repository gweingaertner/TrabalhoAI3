import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

def createRoute(cityList):  # CRIA UMA ROTA ALEATORIA
    route = random.sample(cityList, len(cityList))
    return route

def firstPopulation(popSize, cityfirstlist):  # GERA A POPULAÇÃO
    population = []
    for popRange in range(0, popSize):
        population.append(createRoute(cityfirstlist))
    return population


def routesRank(pop):  # ESCOLHE AS MELHORES ROTAS E RANKEIA ELAS
    calcRes = {}
    for popLen in range(0, len(pop)):
        calcRes[popLen] = Calculator(pop[popLen]).routeCalculator()
    return sorted(calcRes.items(), key=operator.itemgetter(1), reverse=True)


def select_parents(popRanked, bestParent):  # FUNÇÃO QUE SEPARA OS PARES
    selectionResults = []
    distanceFor = pd.DataFrame(np.array(popRanked), columns=["Index", "Distance"])
    distanceFor['cum_sum'] = distanceFor.Distance.cumsum()
    distanceFor['cum_perc'] = 100 * distanceFor.cum_sum / distanceFor.Distance.sum()

    for bestParentRange in range(0, bestParent):
        selectionResults.append(popRanked[bestParentRange][0])
    for popBest in range(0, len(popRanked) - bestParent):
        pick = 100 * random.random()
        for popBest in range(0, len(popRanked)):
            if pick <= distanceFor.iat[popBest, 3]:
                selectionResults.append(popRanked[popBest][0])
                break
    return selectionResults


class Calculator: #CLASSE QUE CALCULA A MENOR DISTANCIA (E INSTANCIA O OBJETO DISTANCIA)
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.bestDistance = 0.0

    def distanceRoute(self):
        if self.distance == 0:
            pathDistance = 0
            for routeFor in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if routeFor + 1 < len(self.route) : toCity = self.route[routeFor + 1]
                else: toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeCalculator(self):
        if self.bestDistance == 0:
            self.bestDistance = 1 / float(self.distanceRoute())
        return self.bestDistance


def uniteCrossoverParents(population, data):  # FUNÇÃO QUE REUNE OS PARES ESCOLHIDOS
    list = []
    for dataFor in range(0, len(data)):
        index = data[dataFor]
        list.append(population[index])
    return list


def crossover(parent1, parent2):  # FUNÇÃO QUE FAZ GERAR DESCENDENTES (CROSSOVER)
    Kid = []
    Parent1 = []
    Parent2 = []
    gene1 = int(random.random() * len(parent1))
    gene2 = int(random.random() * len(parent1))
    startGene = min(gene1, gene2)
    endGene = max(gene1, gene2)
    for geneFor in range(startGene, endGene):
        Parent1.append(parent1[geneFor])
    Parent2 = [item for item in parent2 if item not in Parent1]
    Kid = Parent1 + Parent2
    return Kid


def crossoverPopulation(crossoverData, bestParent):  # EXECUTA O CROSSOVER NA POPULAÇAO
    Kids = []
    length = len(crossoverData) - bestParent
    sample = random.sample(crossoverData, len(crossoverData))

    for bestParentFor in range(0, bestParent):
        Kids.append(crossoverData[bestParentFor])

    for lenghtFor in range(0, length):
        child = crossover(sample[lenghtFor], sample[len(crossoverData) - lenghtFor - 1])
        Kids.append(child)
    return Kids


def mutate(individual, mutationRate):  # EXECUTA A MUTAÇÃO
    for switch in range(len(individual)):
        if (random.random() < mutationRate):
            swapped = int(random.random() * len(individual))

            cityA = individual[switch]
            cityB = individual[swapped]

            individual[switch] = cityB
            individual[swapped] = cityA
    return individual


class Cityclass: #CLASSE DOS OBJETOS CIDADES
    def __init__(self, x_vector, y_vector):
        self.x_vector = x_vector
        self.y_vector = y_vector

    def distance(self, city):
        Xdistance = abs(self.x_vector - city.x_vector)
        Ydistance = abs(self.y_vector - city.y_vector)
        distance = np.sqrt((Xdistance ** 2) + (Ydistance ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x_vector) + "," + str(self.y) + ")"

def mutatePopulation(population, mutationRate):  # EXECUTA O METODO DA MUTAÇÃO EM TODA A POPULAÇÃO (ANTIGA E NOVA)
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, bestParent, mutationRate):  # EXECUTA TODA A MUTAÇÃ0 NAS PROXIMAS GERAÇOES (DO LOOP)
    popSorted = routesRank(currentGen)
    selectionResults = select_parents(popSorted, bestParent)
    uniteCrossoverParentsResult = uniteCrossoverParents(currentGen, selectionResults)
    Kids = crossoverPopulation(uniteCrossoverParentsResult, bestParent)
    mutatedPop = mutatePopulation(Kids, mutationRate)
    return mutatedPop


def geneticAlgorithmAplication(population, populationSize, goodParents, mutationRate, generations):  # APLICA O ALGORITIMO GENETICO
    populatonInstance = firstPopulation(populationSize, population)
    for generationsFor in range(0, generations):
        populatonInstance = nextGeneration(populatonInstance, goodParents, mutationRate)
    bestRouteIndex = routesRank(populatonInstance)[0][0]
    bestRoute = populatonInstance[bestRouteIndex]
    for citycoordenates in bestRoute:
        for cityindex in cityList:
            if cityindex == citycoordenates:
                BestRouteList.append(cityList.index(cityindex) + 1)

    print("Numero de Cidades:" + str(len(cityList)))
    print("População: " + str(SizeOfPopulation_var))
    print("Taxa de Mutação: " + str(Mutation_rate_var))
    print("Melhor Custo: " + str(1 / routesRank(populatonInstance)[0][1]))
    print("Melhor Rota:", end=" ")
    print(BestRouteList)


cityList = []  # LISTA DAS COORDENADAS DAS CIDADES
BestRouteList = []  # LISTA DAS MELHORES ROTAS
SizeOfPopulation_var = 20  # TAMANHO DA POPULAÇÃO
Mutation_rate_var = 0.05  # TAXA DE MUTAÇÃO
Generations_var = 10000  # QUANTIDADE DE GERAÇÕES (O NUMERO DE REPETIÇOES)
BestParentsQtd_var = 10  # NUMEROS DE PAIS DE "ELITE" ESCOLHIDOS

data = np.loadtxt('cidades.mat')  # Carrega O ARQUIVO .MAT
for i in range(0, 20):
    cityList.append(Cityclass(x_vector=data[0][i], y_vector=data[1][i]))

geneticAlgorithmAplication(population=cityList, populationSize=SizeOfPopulation_var, goodParents=BestParentsQtd_var,
                 mutationRate=Mutation_rate_var, generations=Generations_var)  # EXECUTA O ALGORITIMO
