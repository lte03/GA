import cv2
import random
from deap import base,creator,tools
import matplotlib.pyplot as plt

toolbox = base.Toolbox()
image = cv2.imread("target2.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_list = image.tolist()

CXPB = 0.5
MUTPB = 0.2
MAX_ITER = 1000
MAX_POP = 100

def img_flat_gray(img):
    return [pixel for row in img for pixel in row]

def img_unflat_gray(flatten,img_shape):
    w,h = img_shape
    img = [[0 for _ in range(h)] for _ in range(w)]
    k = 0
    for i in range(w):
        for j in range(h):
            img[i][j] = flatten[k]
    return img

def img_flat(img):
    return [color for row in img for pixel in row for color in pixel]

flatten = img_flat_gray(image_list)

def evaluate(individual):
    j = 0
    for i in range(len(flatten)):
        if individual[i] == flatten[i]:
            j+=1
    return -j

creator.create("Fitness",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.Fitness)

toolbox.register("attrs",random.randint,0,255)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attrs, len(flatten))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",evaluate)
toolbox.register("select",tools.selRoulette)
toolbox.register("crossover",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutUniformInt,low=0,up=255,indpb=0.05)

def main():
    pop = toolbox.population(n=MAX_POP)
    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
    print(max(fits))
    while max(fits) < 0 and g < MAX_ITER:
        g = g + 1
        print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Total Population:{len(pop)}")
        best_ind = tools.selBest(pop, 1)[0]
        print(f"Best individual is{best_ind.fitness.values}")
main()

