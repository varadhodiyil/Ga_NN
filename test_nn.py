import numpy as np
from Genetic import Genetic
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


# import the breast cancer dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd


from sklearn.metrics import r2_score

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
label = cancer["target"]
print(label[0])
# print(df.columns)
# df = df.drop(["target"],axis=1)

# splitting the model into training and testing set
X_train, X_test, y_train, y_test = train_test_split(df,
                                                    label, test_size=0.30,
                                                    random_state=101)


def get_model(params):
    print(params)
    classifier = Sequential()

    classifier.add(Dense(units=params[1], bias_initializer='uniform',
                         activation='relu', input_dim=30 ))
    # Adding dropout to prevent overfitting
    classifier.add(Dropout(rate=0.1))
    classifier.add(
        Dense(units=params[1], bias_initializer='uniform', activation='relu'))
    # Adding dropout to prevent overfitting
    classifier.add(Dropout(rate=0.1))
    classifier.add(
        Dense(units=1, bias_initializer='uniform', activation='sigmoid'))
    adam = tf.keras.optimizers.Adam(learning_rate=params[0])

    classifier.compile(
        optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # classifier.summary()
    return classifier


print(X_train.shape[1])
classifier = get_model([0.001 , 16])

# classifier.fit( X_train, y_train,
#     batch_size=100,
#     validation_split=0.2,
#      nb_epoch=30)

# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)


# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))

# print(classifier.evaluate(X_test,y_test) )


# from ga import GA


# X = (X_train.to_numpy() , X_test.to_numpy())
# Y = (y_train, y_test)


# g = GA(n_features=30,n_parents=100, model=classifier, X = X , Y=  Y , metrics= r2_score , model_fn = get_model  )


# chromo, score = g.generations(200,0.10)
# model = get_model(X_train.iloc[:,chromo[-1]].shape[1])
# model.fit(X_train.iloc[:,chromo[-1]],y_train)

# print(model.evaluate(X_test, y_test))


params = [
    {
        'learning_rate': 0.001,
        'min': 0.0001,
        'max': 0.1,
        'low': 0.000001,
        'high': 1,
        'type': float,
    },
    {
        'input_dim': 30,
        'min': 8,
        'max': 16,
        'low': 4,
        'high': 16,
        'type': int,
        'step' : 1
    }
]
g = Genetic(X_train, y_train, X_test, y_test, params, get_model)

g.set_score_fn(r2_score)
numberOfParents = 3  # number of parents to start
numberOfParentsMating = 2  # number of parents that will mate
numberOfParameters = len(params)  # number of parameters that will be optimized
numberOfGenerations = 1  # number of genration that will be created

# define the population size


best = g.train(3,2,1)
print(best)

# # Best solution from the final iteration
# g.set_population(population)
# fitness = g.train_population()
# fitnessHistory[generation+1, :] = fitness

# # index of the best solution
# bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

# # Best fitness
# print("Best fitness is =", fitness[bestFitnessIndex])

# # Best parameters
# print("Best parameters are:")
# print('learning_rate', population[bestFitnessIndex])
