import numpy as np
import math
import scipy.stats as st


def calculate_rmse(measured_mos, predicted_mos):
    number_of_samples = measured_mos.shape[0]
    perror = measured_mos - predicted_mos
    perror_sum = np.sum((perror**2), axis=0)
    rmse = math.sqrt(perror_sum / (number_of_samples - 1))
    return rmse


def confidence_interval_rmse(rmse, n, number_of_samples):
    lower_bound = rmse*math.sqrt(n) / math.sqrt(st.chi2.interval(alpha=0.95, df=(number_of_samples-4))[1])
    upper_bound = rmse*math.sqrt(n) / math.sqrt(st.chi2.interval(alpha=0.95, df=(number_of_samples-4))[0])
    interval = np.vstack([lower_bound, upper_bound])
    return interval


def compare_rmse(rmse_1, rmse_2, threshold):
    if rmse_1 > rmse_2:
        q = math.pow(rmse_1, 2) / math.pow(rmse_2, 2)
        if q >= threshold:
            return -1   # worse
        else:
            return 0    # same
    else:
        q = math.pow(rmse_2, 2) / math.pow(rmse_1, 2)
        if q >= threshold:
            return 1    # better
        else:
            return 0    # same


def compare_models(rmse_param, threshold, number_of_metrics):
    compared = np.zeros((number_of_metrics, number_of_metrics))
    for x in range(0, number_of_metrics):
        for y in range(0, number_of_metrics):
            if x < y:
                comparing_result = compare_rmse(rmse_param[x], rmse_param[y], threshold)
                compared[x, y] = comparing_result
    return compared


def get_rmse(measured_mos, predicted_mos, all_rmse, all_rmse_ci, all_rmse_compare_arr):
    # RMSE
    number_of_samples = predicted_mos.shape[1]
    number_of_metrics = predicted_mos.shape[2]

    rmse_arr = []
    rmse_ci_arr = []
    n = number_of_samples - 4
    for x in range(number_of_metrics):
        rmse_arr.append(calculate_rmse(measured_mos, predicted_mos[0, :, x]))
        rmse_ci_arr.append(confidence_interval_rmse(rmse_arr[-1], n, number_of_samples))
    rmse = np.hstack(rmse_arr)
    rmse_ci = np.hstack(rmse_ci_arr)

    rmse_threshold = st.f.ppf(0.95, (number_of_samples - 4), (number_of_samples - 1))
    rmse_compared = compare_models(rmse, rmse_threshold, number_of_metrics)

    all_rmse.append(rmse)
    all_rmse_ci.append(rmse_ci)
    all_rmse_compare_arr.append(rmse_compared)

    return all_rmse, all_rmse_ci, all_rmse_compare_arr
