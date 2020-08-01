#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
import numpy as np
import random

from datetime import datetime
from tensorflow import keras
random.seed(42)
np.random.seed(42)
import tensorflow as tf

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)



class Genetic():
    def __init__(self, X_train, Y_train, X_test, Y_test, params, model_fn):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.hyper_params = params
        self.tuneable_params = list()
        self.score_fn = None
        self.model_fn = model_fn

    def set_population(self, p):
        self.population = p

    def init_population(self, parents=8):
        for i, e in enumerate(self.hyper_params):
            self.tuneable_params.append(
                np.empty([parents, 1], dtype=e['type']))
            for j in range(parents):
                if 'step' in e:
                    self.tuneable_params[i][j] = random.randrange(
                        e['min'], e['max'], step=e['step'])
                else:
                    self.tuneable_params[i][j] = random.uniform(
                        e['min'], e['max'])
        self.population = np.asarray(self.tuneable_params)
        self.population = np.concatenate(self.population, axis=1)
        return self.population

    def set_score_fn(self, score_fn):
        self.score_fn = score_fn

    def score(self, y_pred):
        if self.score_fn is None:
            self.score_fn = f1_score
        return self.score_fn(y_pred, self.Y_test)

    # train the data annd find fitness score

    def train__population(self):
        fScore = []
        for i in range(self.population.shape[0]):
            print('param', self.population[i][0])
            model = self.model_fn(self.population[i])
            model.fit(self.X_train, self.Y_train, batch_size=10, epochs=5 , validation_size=0.2, callbacks=[tensorboard_callback])
            preds = model.predict(self.X_test)
            preds = preds > 0.5
            fScore.append(self.score(preds))
        return fScore

    # Survial of fittest

    def form_parents(self, population, fitness_scores, n_parents):
        selectedParents = np.empty((n_parents, population.shape[1]))

        for i in range(n_parents):
            max_score = np.where(fitness_scores == np.max(fitness_scores))
            max_score = max_score[0][0]
            selectedParents[i, :] = population[max_score, :]
            fitness_scores[max_score] = -1
        return selectedParents


    
    def crossover_uniform(self, parents, childrenSize):

        crossoverPointIndex = np.arange(0, np.uint8(
            childrenSize[1]), 1, dtype=np.uint8)  # get all the index
        crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]), np.uint8(
            childrenSize[1]/2))  # select half  of the indexes randomly
        crossoverPointIndex2 = np.array(list(
            set(crossoverPointIndex) - set(crossoverPointIndex1)))  # select leftover indexes

        children = np.empty(childrenSize)

        for i in range(childrenSize[0]):

            # find parent 1 index
            parent1_index = i % parents.shape[0]
            # find parent 2 index
            parent2_index = (i+1) % parents.shape[0]
            # insert parameters based on random selected indexes in parent 1
            children[i, crossoverPointIndex1] = parents[parent1_index,
                                                        crossoverPointIndex1]
            # insert parameters based on random selected indexes in parent 1
            children[i, crossoverPointIndex2] = parents[parent2_index,
                                                        crossoverPointIndex2]
        return children

    def mutation(self, crossover, n_params):
        # Define minimum and maximum values allowed for each parameter

        m_value = np.zeros((n_params, 2))
        for i, e in enumerate(self.hyper_params):
            m_value[i, :] = [e['min'], e['max']]

        mutationValue = 0
        rand_mutate = np.random.randint(0, len(self.hyper_params), 1)[0]
        mutationValue = round(np.random.uniform(
            self.hyper_params[rand_mutate]['low'], self.hyper_params[rand_mutate]['high']), 2)

        for i in range(crossover.shape[0]):
            crossover[i, rand_mutate] = crossover[i,
                                                  rand_mutate] + mutationValue
            if(crossover[i, rand_mutate] > m_value[rand_mutate, 1]):
                crossover[i, rand_mutate] = m_value[rand_mutate, 1]
            if(crossover[i, rand_mutate] < m_value[rand_mutate, 0]):
                crossover[i, rand_mutate] = m_value[rand_mutate, 0]
        return crossover


    


    def train(self, n_parents=3, min_parent_mating=2, n_generation=1):
        assert min_parent_mating >= 2
        assert n_generation > 0
        assert n_parents >= 3
        n_params = len(self.hyper_params)
        population_size = (n_parents, n_params)

        population = self.init_population(n_parents)

        # define an array to store the fitness  hitory
        fitnessHistory = np.empty([n_generation + 1, n_parents])

        populationHistory = np.empty(
            [(n_generation+1)*n_parents, n_params])

        populationHistory[0:n_parents, :] = population

        for generation in range(n_parents):
            print("This is number %s generation" % (generation))

            # train the dataset and obtain fitness
            fitnessValue = self.train__population()
            fitnessHistory[generation, :] = fitnessValue

            # best score in the current iteration
            print('Best max in the this iteration = {}'.format(
                np.max(fitnessHistory[generation, :])))

            # survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
            parents = self.form_parents(
                population=population, fitness_scores=fitnessValue, n_parents=min_parent_mating)

            # mate these parents to create children having parameters from these parents (we are using uniform crossover)
            children = self.crossover_uniform(parents=parents, childrenSize=(
                population_size[0] - parents.shape[0], n_params))

            children_mutated = self.mutation(children, n_params)

            population[0:parents.shape[0], :] = parents  # fittest parents
            population[parents.shape[0]:, :] = children_mutated  # children

            populationHistory[(generation+1)*n_parents: (generation+1)*n_parents +
                              n_parents, :] = population  # srore parent information

            self.set_population(population)
            fitness = self.train__population()
            fitnessHistory[generation+1, :] = fitness

            # index of the best solution
            best = np.where(fitness == np.max(fitness))[0][0]

            # Best fitness
            print("Best fitness is =", population[best])
            return population[best]


if __name__ == "__main__":
    params = [
        {
            'learning_rate': 0.001,
            'min': 0.0001,
            'max': 0.1,
            'low': 0.000001,
            'high': 1,
            'type': np.float,
        },
        {
            'input_dim': 30,
            'min': 30,
            'max': 64,
            'low': 20,
            'high': 64,
            'type': np.uint8
        }
    ]
    g = Genetic([], [], [], [], params, None)
    print(g.init_population())
