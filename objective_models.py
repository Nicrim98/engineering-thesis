from generate_data import *

acr = [1, 2, 3, 4, 5]


def addNoise(data, mean, st_deviation, size):
    model = data + normal(mean, st_deviation, size)
    model[model > acr[-1]] = acr[-1]
    model[model < acr[0]] = acr[0]
    return model


def loadModels(psi, number_of_samples):
    models = []
    predictedMOS_best = psi
    models.append(predictedMOS_best)
    predictedMOS_good = addNoise(psi, 0, 0.25, number_of_samples)
    models.append(predictedMOS_good)
    predictedMOS_bad = addNoise(psi, 0, 0.5, number_of_samples)
    models.append(predictedMOS_bad)
    predictedMOS_worse = addNoise(psi, 0, 0.75, number_of_samples)
    models.append(predictedMOS_worse)
    predictedMOS_the_worst = addNoise(psi, 0, 1, number_of_samples)
    models.append(predictedMOS_the_worst)
    return models
