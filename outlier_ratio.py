import numpy as np
import math
import scipy.stats as st


def confidence_interval_outlier_ratio(outlier_ratio, number_of_samples):
    z = 1.96
    outlier_ratio_st_deviation = math.sqrt((outlier_ratio*(1-outlier_ratio)) / number_of_samples)
    lower_bound = outlier_ratio - outlier_ratio_st_deviation * z
    upper_bound = outlier_ratio + outlier_ratio_st_deviation * z
    interval = np.vstack([lower_bound, upper_bound])
    return interval


def calculate_outlier_confidence_interval(ratings):
    # potrzebna inna nazwa do tego !!!
    number_voters = ratings.shape[1]
    z = 1.96
    confidence_interval = (z * np.std(ratings, axis=1)) / math.sqrt(number_voters)
    return confidence_interval


def calculate_outlier_ratio(measured_mos, predicted_mos, confidence_interval, number_of_samples):
    points = confidence_interval - (measured_mos - predicted_mos)**2
    outliers = [out for out in points if out < 0]
    outlier_ratio = len(outliers) / number_of_samples
    return outlier_ratio


def compare_outlier_ratio(outlier_1, outlier_2, threshold, number_of_samples):
    if outlier_1 == outlier_2:
        return 0
    else:
        p = ((number_of_samples*outlier_1) + (number_of_samples*outlier_2)) / (2*number_of_samples)
        outlier_ratio_std = math.sqrt(p * (1 - p) * ((1 / number_of_samples) + (1 / number_of_samples)))
        zn = (outlier_1 - outlier_2) / outlier_ratio_std
        if abs(zn) <= threshold:
            return 0    # same
        else:   # statistically different
            if outlier_1 < outlier_2:
                return 1
            else:
                return -1


def compare_models(outlier_param, threshold, number_of_metrics, number_of_samples):
    compared = np.zeros((number_of_metrics, number_of_metrics))
    for x in range(0, number_of_metrics):
        for y in range(0, number_of_metrics):
            if x < y:
                comparing_result = compare_outlier_ratio(outlier_param[x], outlier_param[y], threshold,
                                                         number_of_samples)
                compared[x, y] = comparing_result
    return compared


def get_outlier(measured_mos, predicted_mos, ratings, all_outlier, all_outlier_ratio_ci, all_outlier_ratio_compare_arr):
    # Outlier Ratio
    number_of_samples = predicted_mos.shape[1]
    number_of_metrics = predicted_mos.shape[2]

    confidence_interval = calculate_outlier_confidence_interval(ratings)
    outlier_ratio_arr = []
    outlier_ratio_confidence_arr = []
    for x in range(number_of_metrics):
        outlier_ratio_arr.append(calculate_outlier_ratio(measured_mos, predicted_mos[0, :, x], confidence_interval,
                                                         number_of_samples))
        outlier_ratio_confidence_arr.append(confidence_interval_outlier_ratio(outlier_ratio_arr[-1], number_of_samples))
    outlier_ratio = np.hstack(outlier_ratio_arr)
    outlier_ratio_ci = np.hstack(outlier_ratio_confidence_arr)

    t_student_threshold = st.t.ppf(0.95, (number_of_samples - 4))
    outlier_ratio_compared = compare_models(outlier_ratio, t_student_threshold, number_of_metrics, number_of_samples)

    all_outlier.append(outlier_ratio)
    all_outlier_ratio_ci.append(outlier_ratio_ci)
    all_outlier_ratio_compare_arr.append(outlier_ratio_compared)

    return all_outlier, all_outlier_ratio_ci, all_outlier_ratio_compare_arr
