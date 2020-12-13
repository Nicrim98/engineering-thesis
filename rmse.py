import numpy as np
import math
import scipy.stats as st


def calculateRMSE(measuredMOS, predictedMOS):
    number_of_samples = measuredMOS.shape[0]
    perror = measuredMOS - predictedMOS
    perror_sum = np.sum((perror**2), axis=0)
    rmse = math.sqrt( perror_sum / (number_of_samples - 1))
    return rmse


def confidenceIntervalRMSE(rmse, n, number_of_samples):
    lower = rmse*math.sqrt(n) / math.sqrt(st.chi2.interval(alpha=0.95, df=(number_of_samples-4))[1])
    upper = rmse*math.sqrt(n) / math.sqrt(st.chi2.interval(alpha=0.95, df=(number_of_samples-4))[0])
    interval = np.vstack([lower, upper])
    return interval


def compareRMSE(rmse1, rmse2, threshold):
    if rmse1 > rmse2:
        q = math.pow(rmse1, 2) / math.pow(rmse2, 2)
        if q >= threshold:
            return -1   # worse
        else:
            return 0 # same
    else:
        q = math.pow(rmse2, 2) / math.pow(rmse1, 2)
        if q >= threshold:
            return 1    # better
        else:
            return 0    # same


def compareModels(rmse_param, threshold, number_of_metrics):
    compared = np.zeros((number_of_metrics, number_of_metrics))
    for x in range(0, number_of_metrics):
        for y in range(0, number_of_metrics):
            if x < y:
                comparing_result = compareRMSE(rmse_param[x], rmse_param[y], threshold)
                compared[x, y] = comparing_result
    return compared


def getRMSE(measuredMOS, predictedMOS, all_rmse, all_rmse_CI, all_rmse_compare_arr):
    # RMSE
    number_of_samples = predictedMOS.shape[1]
    number_of_metrics = predictedMOS.shape[2]

    rmse_arr = []
    rmse_CI_arr = []
    n = number_of_samples - 4
    for x in range(number_of_metrics):
        rmse_arr.append(calculateRMSE(measuredMOS, predictedMOS[0, :, x]))
        rmse_CI_arr.append(confidenceIntervalRMSE(rmse_arr[-1], n, number_of_samples))
    rmse = np.hstack(rmse_arr)
    rmse_CI = np.hstack(rmse_CI_arr)

    rmse_threshold = st.f.ppf(0.95, (number_of_samples - 4), (number_of_samples - 1))
    rmseCompared = compareModels(rmse, rmse_threshold, number_of_metrics)

    all_rmse.append(rmse)
    all_rmse_CI.append(rmse_CI)
    all_rmse_compare_arr.append(rmseCompared)

    return all_rmse, all_rmse_CI, all_rmse_compare_arr