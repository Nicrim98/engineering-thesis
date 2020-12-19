import numpy as np
import _pickle as cpickle
import bz2

number_of_voters = 30
number_of_sequences = 100
acr = [1, 2, 3, 4, 5]
psi_low, psi_high = 1.0, 5.0
delta_mean, delta_st_deviation = 0, 0
epsilon_low, epsilon_high = 0, 0


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cpickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cpickle.load(data)
    return data


def uniform(low, high, size):
    return np.random.uniform(low, high, size)


def normal(mean, st_deviation, size):
    return np.random.normal(loc=mean, scale=st_deviation, size=size)


def scoring_process(psi, delta, epsilon):
    ones = np.ones(number_of_voters)
    extended_psi = np.outer(psi, ones)
    ratings = np.round(normal(mean=(extended_psi + delta), st_deviation=epsilon, size=(number_of_sequences,
                                                                                       number_of_voters)))
    ratings[ratings > acr[-1]] = acr[-1]
    ratings[ratings < acr[0]] = acr[0]
    measured_mos = np.mean(ratings, axis=1)
    return ratings, measured_mos


def generate_subjective_data(number):
    # generowanie liczb opisujących jakość (psi)
    psi = uniform(psi_low, psi_high, number_of_sequences)
    # generowanie liczb opisujących obciążenie (delta)
    delta = normal(delta_mean, delta_st_deviation, number_of_voters)
    # generowanie liczb opisujących stabilność odpowiedzi (sigma)
    epsilon = uniform(epsilon_low, epsilon_high, number_of_voters)
    # generate subjective scoring assesment
    ratings, measured_mos = scoring_process(psi, delta, epsilon)

    # compressed_pickle('input_data/psi/psi_'+str(number), psi)
    # compressed_pickle('input_data/delta/delta_'+str(number), delta)
    # compressed_pickle('input_data/epsilon/epsilon_'+str(number), epsilon)
    # compressed_pickle('input_data/ratings/ratings_' + str(number), ratings)
    return psi, delta, epsilon, ratings, measured_mos
