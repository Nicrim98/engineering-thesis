import numpy as np
import math
import scipy.stats as st


def confidenceIntervalPearsonCoefficient(R, number_of_samples):
    z = 0.5 * math.log((1 + R) / (1 - R))
    pearson_st_deviation = 1 / math.sqrt(number_of_samples - 3)
    lower = np.tanh(z - (1.96 * pearson_st_deviation))
    upper = np.tanh(z + (1.96 * pearson_st_deviation))
    interval = np.vstack([lower, upper])
    return interval


def calculatePearsonCoefficient(measuredMOS, predictedMOS, mean_of_est_PVSs, mean_of_metric_PVSs):
    numerator_pearson = np.sum(((measuredMOS - mean_of_est_PVSs) * (predictedMOS - mean_of_metric_PVSs)), axis=0)
    denominator_pearson = math.sqrt(np.sum(((measuredMOS - mean_of_est_PVSs) ** 2), axis=0)) * math.sqrt(np.sum(((predictedMOS - mean_of_metric_PVSs) ** 2), axis=0))
    pearson_correlation_coef = numerator_pearson / denominator_pearson
    return pearson_correlation_coef


def comparePearsonCoeff(pearson1, pearson2, standard_deviation, threshold):
    z1 = 0.5 * math.log((1 + pearson1) / (1 - pearson1))
    z2 = 0.5 * math.log((1 + pearson2) / (1 - pearson2))
    zn = (z1 - z2) / standard_deviation
    if abs(zn) <= threshold:
        return 0  # same
    else:  # statistically different
        if pearson1 > pearson2:
            return 1
        else:
            return -1


def compareModels(pearson_param, threshold, number_of_metrics, zn_standard_dev):
    compared = np.zeros((number_of_metrics, number_of_metrics))
    for x in range(0, number_of_metrics):
        for y in range(0, number_of_metrics):
            if x < y:
                comparing_result = comparePearsonCoeff(pearson_param[x], pearson_param[y], zn_standard_dev, threshold)
                compared[x, y] = comparing_result
    return compared

def getPearson(measuredMOS, predictedMOS, all_pearson, all_pearson_CI, all_pearson_compare_arr):
    # PEARSON CORRELATION COEFFICIENT R
    number_of_samples = predictedMOS.shape[1]
    number_of_metrics = predictedMOS.shape[2]

    pearson_coeff_arr = []
    pearson_coeff_CI_arr = []
    mean_of_est_PVSs = np.mean(measuredMOS, axis=0)
    for x in range(number_of_metrics):
        mean_of_metric_PVSs = np.mean(predictedMOS[0, :, x], axis=0)
        pearson_coeff_arr.append(calculatePearsonCoefficient(measuredMOS, predictedMOS[0, :, x], mean_of_est_PVSs, mean_of_metric_PVSs))
        pearson_coeff_CI_arr.append(confidenceIntervalPearsonCoefficient(pearson_coeff_arr[-1], number_of_samples))
    pearson_coeff = np.hstack(pearson_coeff_arr)
    pearson_coeff_CI = np.hstack(pearson_coeff_CI_arr)

    zn_std_tmp = 1 / math.sqrt(number_of_samples - 3)
    zn_std = math.sqrt( math.pow(zn_std_tmp, 2) + math.pow(zn_std_tmp, 2) )
    t_student_threshold = st.t.ppf(0.95, (number_of_samples - 4))
    pearson_compared = compareModels(pearson_coeff, t_student_threshold, number_of_metrics, zn_std)

    all_pearson.append(pearson_coeff)
    all_pearson_CI.append(pearson_coeff_CI)
    all_pearson_compare_arr.append(pearson_compared)

    return all_pearson, all_pearson_CI, all_pearson_compare_arr
