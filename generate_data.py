import numpy as np

number_of_voters = 30
number_of_sequences = 100
acr = [1, 2, 3, 4, 5]
psi_low, psi_high = 1.0, 5.0
delta_mean, delta_st_deviation = 0, 0.7
epsilon_low, epsilon_high = 0.3, 0.9


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


def generate_subjective_data():
    # sequence quality
    psi = uniform(psi_low, psi_high, number_of_sequences)
    # user imprecision
    delta = normal(delta_mean, delta_st_deviation, number_of_voters)
    # error
    epsilon = uniform(epsilon_low, epsilon_high, number_of_voters)
    # scored sequenced
    ratings, measured_mos = scoring_process(psi, delta, epsilon)
    return psi, delta, epsilon, ratings, measured_mos
