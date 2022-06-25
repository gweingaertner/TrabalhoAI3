import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

class Calculator: #CLASSE QUE CALCULA A MENOR DISTANCIA
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def distanceRoute(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeCalculator(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.distanceRoute())
        return self.fitness

def createRoute(cityList):  # CRIA UMA ROTA ALEATORIA
    route = random.sample(cityList, len(cityList))
    return route

class Cityclass: #CLASSE DOS OBJETOS CIDADES
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        Xdistance = abs(self.x - city.x)
        Ydistance = abs(self.y - city.y)
        distance = np.sqrt((Xdistance ** 2) + (Ydistance ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def firstPopulation(popSize, cityList):  # GERA A POPULAÇÃO
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def routesRank(population):  # ESCOLHE AS MELHORES ROTAS E RANKEIA ELAS
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Calculator(population[i]).routeCalculator()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):  # FUNÇÃO QUE SEPARA OS PARES
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):  # FUNÇÃO QUE REUNE OS PARES ESCOLHIDOS
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):  # FUNÇÃO QUE FAZ GERAR DESCENDENTES (CROSSOVER)
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):  # EXECUTA O CROSSOVER NA POPULAÇAO
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):  # EXECUTA A MUTAÇÃO
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):  # EXECUTA O METODO DA MUTAÇÃO EM TODA A POPULAÇÃO (ANTIGA E NOVA)
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):  # EXECUTA TODA A MUTAÇÃ0 NAS PROXIMAS GERAÇOES (DO LOOP)
    popRanked = routesRank(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, populationSize, goodParents, mutationRate, generations):  # APLICA O ALGORITIMO GENETICO
    pop = firstPopulation(populationSize, population)
    for i in range(0, generations):
        pop = nextGeneration(pop, goodParents, mutationRate)
    bestRouteIndex = routesRank(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    for citycoordenates in bestRoute:
        for cityindex in cityList:
            if cityindex == citycoordenates:
                BestRouteList.append(cityList.index(cityindex) + 1)

    print("Numero de Cidades:" + str(len(cityList)))
    print("População: " + str(SizeOfPopulation_var))
    print("Taxa de Mutação: " + str(Mutation_rate_var))
    print("Melhor Custo: " + str(1 / routesRank(pop)[0][1]))
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
    cityList.append(Cityclass(x=data[0][i], y=data[1][i]))

geneticAlgorithm(population=cityList, populationSize=SizeOfPopulation_var, goodParents=BestParentsQtd_var,
                 mutationRate=Mutation_rate_var, generations=Generations_var)  # EXECUTA O ALGORITIMO
