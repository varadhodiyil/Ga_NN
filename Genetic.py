#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

random.seed(42)
np.random.seed(42)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
writer = tf.summary.create_file_writer(logdir)


class Genetic():
    def __init__(self, X_train, Y_train, X_test, Y_test, params, model_fn , eval_fn=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.hyper_params = params
        self.tuneable_params = list()
        self.score_fn = None
        self.model_fn = model_fn
        self.eval_fn = eval_fn

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
        print(self.population)
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
            
            # model.fit([X_train_tokens,Y_train_tokens[:,:-1]], Y_train_tokens.reshape(Y_train_tokens.shape[0],Y_train_tokens.shape[1], 1)[:,1:] ,epochs=200,batch_size=64, validation_split=0.2)
            # model.fit([self.X_train,self.Y_train[:,:-1]], \
            #                 self.Y_train.reshape(self.Y_train.shape[0],self.Y_train.shape[1], 1)[:,1:] , 
            #                 epochs=5,batch_size=64, validation_split=0.2)
            model.fit(self.X_train , self.Y_train , validation_split=0.2 , epochs=10)
            if not self.eval_fn:
                preds = model.predict(self.X_test)
                preds = preds > 0.5
            else:
                preds = self.eval_fn(self.X_test)
            fScore.append(self.score(preds))

        self.curr_score = np.asarray(fScore)
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
            [(n_generation+1)*n_parents , n_params])
        print("History Shape",populationHistory.shape)

        populationHistory[0:n_parents, :] = population

        for generation in range(n_generation):
            self.curr_gen = generation
            print("This is number %s generation" % (generation))

            # train the dataset and obtain fitness
            fitnessValue = self.train__population()
            fitnessHistory[generation, :] = fitnessValue

            # best score in the current iteration
            print('Best max in the this iteration = {}'.format(
                np.max(fitnessHistory[generation, :])))
            
            with writer.as_default():
                tf.summary.scalar("generation {0}".format(generation), self.curr_score.mean(), step=generation)

            # survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
            parents = self.form_parents(
                population=population, fitness_scores=fitnessValue, n_parents=min_parent_mating)

            # mate these parents to create children having parameters from these parents (we are using uniform crossover)
            children = self.crossover_uniform(parents=parents, childrenSize=(
                population_size[0] - parents.shape[0], n_params))

            children_mutated = self.mutation(children, n_params)

            population[0:parents.shape[0], :] = parents  # fittest parents
            population[parents.shape[0]:, :] = children_mutated  # children
            print(population)
            print(populationHistory)
            print((generation+1)*n_parents , (generation+1)*n_parents + n_parents)
            
            populationHistory[(generation+1)*n_parents : (generation+1)*n_parents+ n_parents , :] = population 
            
            tf.keras.backend.clear_session()


        self.set_population(population)
        fitness = self.train__population()
        fitnessHistory[generation+1, :] = fitness

        # index of the best solution
        best = np.where(fitness == np.max(fitness))[0][0]

        # Best fitness
        print("Best fitness is =", population[best])
        self.plot_parameters(n_generation, n_parents, fitnessHistory, "fitness (F1-score)")
        return population[best]


    def plot_parameters(self , numberOfGenerations, numberOfParents, parameter, parameterName):
        #inspired from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
        generationList = ["Gen {}".format(i) for i in range(numberOfGenerations+1)]
        populationList = ["Parent {}".format(i) for i in range(numberOfParents)]
        
        fig, ax = plt.subplots()
        im = ax.imshow(parameter, cmap=plt.get_cmap('YlOrBr'))
        
        # show ticks
        ax.set_xticks(np.arange(len(populationList)))
        ax.set_yticks(np.arange(len(generationList)))
        
        # show labels
        ax.set_xticklabels(populationList)
        ax.set_yticklabels(generationList)
        
        # set ticks at 45 degrees and rotate around anchor
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        
        
        # insert the value of the parameter in each cell
        for i in range(len(generationList)):
            for j in range(len(populationList)):
                text = ax.text(j, i, parameter[i, j],
                            ha="center", va="center", color="k")
        
        ax.set_title("Change in the value of " + parameterName)
        fig.tight_layout()
        plt.show()


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
