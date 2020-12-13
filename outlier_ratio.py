import numpy as np
import math
import scipy.stats as st


def confidenceIntervalOutlierRatio(outlier_ratio, number_of_samples):
    z = 1.96
    outlier_ratio_st_deviation = math.sqrt( (outlier_ratio*(1-outlier_ratio)) / number_of_samples )
    lower = outlier_ratio - outlier_ratio_st_deviation * z
    upper = outlier_ratio + outlier_ratio_st_deviation * z
    interval = np.vstack([lower, upper])
    return interval


def calculateOutlierConfidenceInterval(ratings):
    # potrzebna inna nazwa do tego !!!
    number_voters = ratings.shape[1]
    z = 1.96
    numerator_or = z * np.std(ratings, axis=1)
    denominator_or = math.sqrt(number_voters)
    confidenceInterval = numerator_or / denominator_or
    return confidenceInterval


def calculateOutlierRatio(measuredMOS, predictedMOS, confidenceInterval, number_of_samples):
    points = confidenceInterval - (measuredMOS - predictedMOS)**2
    outliers = [out for out in points if out < 0]
    outlierRatio = len(outliers) / number_of_samples
    return outlierRatio



def compareOutlierRatio(outlier1, outlier2, threshold, number_of_samples):
    if outlier1 == outlier2:
        return 0
    else:
        p = ( (number_of_samples*outlier1) + (number_of_samples*outlier2) ) / (2*number_of_samples)
        outlier_ratio_std = math.sqrt(p * (1 - p) * ( (1 / number_of_samples) + (1 / number_of_samples) ))
        zn = (outlier1 - outlier2) / outlier_ratio_std
        if abs(zn) <= threshold:
            return 0    # same
        else:   # statistically different
            if outlier1 < outlier2:
                return 1
            else:
                return -1


def compareModels(outlier_param, threshold, number_of_metrics, number_of_samples):
    compared = np.zeros((number_of_metrics, number_of_metrics))
    for x in range(0, number_of_metrics):
        for y in range(0, number_of_metrics):
            if x < y:
                comparing_result = compareOutlierRatio(outlier_param[x], outlier_param[y], threshold, number_of_samples)
                compared[x, y] = comparing_result
    return compared


def getOutlier(measuredMOS, predictedMOS, ratings, all_outlier, all_outlier_ratio_CI, all_OutlierRatio_compare_arr):
    # Outlier Ratio
    number_of_samples = predictedMOS.shape[1]
    number_of_metrics = predictedMOS.shape[2]

    confidenceInterval = calculateOutlierConfidenceInterval(ratings)
    outlier_ratio_arr = []
    outlier_ratio_confidence_arr = []
    for x in range(number_of_metrics):
        outlier_ratio_arr.append(calculateOutlierRatio(measuredMOS, predictedMOS[0, :, x], confidenceInterval, number_of_samples))
        outlier_ratio_confidence_arr.append(confidenceIntervalOutlierRatio(outlier_ratio_arr[-1], number_of_samples))
    outlier_ratio = np.hstack(outlier_ratio_arr)
    outlier_ratio_CI = np.hstack(outlier_ratio_confidence_arr)

    t_student_threshold = st.t.ppf(0.95, (number_of_samples - 4))
    outlierRatio_compared = compareModels(outlier_ratio, t_student_threshold, number_of_metrics, number_of_samples)

    all_outlier.append(outlier_ratio)
    all_outlier_ratio_CI.append(outlier_ratio_CI)
    all_OutlierRatio_compare_arr.append(outlierRatio_compared)

    return all_outlier, all_outlier_ratio_CI, all_OutlierRatio_compare_arr
