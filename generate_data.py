import numpy as np
import _pickle as cPickle
import bz2

acr = [1, 2, 3, 4, 5]
psi_low, psi_high, psi_size = 1.0, 5.0, 100
delta_mean, delta_st_deviation, delta_size = 0.5, 1, 30
epsilon_low, epsilon_high, epsilon_size = 0.7, 0.9, 30


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def uniform(low, high, size):
    return np.random.uniform(low, high, size)


def normal(mean, st_deviation, size):
    return np.random.normal(loc=mean, scale=st_deviation, size=size)


def generateSubjectiveData(number):
    # generowanie liczb opisujących jakość (psi)
    psi = uniform(psi_low, psi_high, psi_size)

    # generowanie liczb opisujących obciążenie (delta)
    delta = normal(delta_mean, delta_st_deviation, delta_size)

    # generowanie liczb opisujących stabilność odpowiedzi (sigma)
    epsilon = uniform(epsilon_low, epsilon_high, epsilon_size)

    # generate subjective scoring assesment
    ones = np.ones(delta_size)
    extended_psi = np.outer(psi, ones)
    ratings = np.round(normal(mean=(extended_psi + delta), st_deviation=epsilon, size=(psi_size, delta_size)))
    # elimnacja ocen powyżej skali
    ratings[ratings > acr[-1]] = acr[-1]
    ratings[ratings < acr[0]] = acr[0]

    #compressed_pickle('input_data/psi/psi_'+str(number), psi)
    #compressed_pickle('input_data/delta/delta_'+str(number), delta)
    #compressed_pickle('input_data/epsilon/epsilon_'+str(number), epsilon)
    #compressed_pickle('input_data/ratings/ratings_' + str(number), ratings)

    measuredMOS = np.mean(ratings, axis=1)

    return psi, delta, epsilon, ratings, measuredMOS


def only_scoring_process(psi, delta, epsilon):
    ones = np.ones(delta_size)
    extended_psi = np.outer(psi, ones)
    ratings = np.round(normal(mean=(extended_psi + delta), st_deviation=epsilon, size=(psi_size, delta_size)))
    # elimnacja ocen wychodzących powyżej skali ACR 1-5
    ratings[ratings > acr[-1]] = acr[-1]
    ratings[ratings < acr[0]] = acr[0]

    return ratings