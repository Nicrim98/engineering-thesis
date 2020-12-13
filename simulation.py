from generate_data import compressed_pickle, decompress_pickle, generateSubjectiveData, normal, only_scoring_process
from objective_models import *
import numpy as np
import time
import rmse as rmse
import outlier_ratio as outlier
import pearson as pearson
from algorithms_comparison import compare_algorithms

sim_loop = 1000
acr = [1, 2, 3, 4, 5]
all_MOS_measured = []

all_psi_arr = []
all_delta_arr = []
all_epsilon_arr = []
all_ratings_arr = []

all_rmse = []
all_outlier = []
all_pearson = []

all_rmse_CI = []
all_outlier_CI = []
all_pearson_CI = []

all_rmse_compare_arr = []
all_pearson_compare_arr = []
all_OutlierRatio_compare_arr = []

params = []


def includeParamInComparison(param_name, all_values, all_values_compare_arr, algorithms):
    all_values = np.vstack(all_values)
    all_values_compared = np.stack(all_values_compare_arr, axis=2)
    algorithms.append(all_values_compared)
    compressed_pickle('all_'+param_name+'_compared', all_values_compared)
    return all_values, all_values_compared, algorithms



#psi, delta, epsilon, ratings = generateData(0)

start_time = time.time()
for number in range(0, sim_loop, 1):
    print(number)
    psi, delta, epsilon, ratings, measuredMOS = generateSubjectiveData(number)

    number_of_samples = ratings.shape[0]
    number_voters = ratings.shape[1]

    # OBJECTIVE MODELS
    predictedMOS_arr = loadModels(psi, number_of_samples)
    predictedMOS = np.dstack(predictedMOS_arr)
    number_of_metrics = predictedMOS.shape[2]


    # METRICS
    rmse.getRMSE(measuredMOS, predictedMOS, all_rmse, all_rmse_CI, all_rmse_compare_arr)
    outlier.getOutlier(measuredMOS, predictedMOS, ratings, all_outlier, all_outlier_CI, all_OutlierRatio_compare_arr)
    pearson.getPearson(measuredMOS, predictedMOS, all_pearson, all_pearson_CI, all_pearson_compare_arr)


    # ### Liczenie R^2 dla psi lepiej działa niż peaeson correlation coefficient XD
    # r_squared = []
    # for met in range(0, number_of_metrics):
    #     mean_score_PVS_tmp = np.mean(stacked_all_metrics[0, :, met], axis=0)
    #     ones_1 = np.ones(number_of_samples)
    #     mean_score_PVS = np.outer(mean_score_PVS_tmp, ones_1)
    #     licznik = np.sum(np.subtract(MOS_measured, mean_score_PVS) * np.subtract(MOS_measured, mean_score_PVS))
    #     mianownik = np.sum(np.subtract(stacked_all_metrics[0, :, met], mean_score_PVS) * np.subtract(stacked_all_metrics[0, :, met], mean_score_PVS))
    #     wynik_psi = licznik / mianownik
    #     r_squared.append(wynik_psi)
    #
    #     r_squared_final = np.hstack(r_squared)

    # wartości zawsze występujące w mojej symulacji
    all_psi_arr.append(psi)
    all_delta_arr.append(delta)
    all_epsilon_arr.append(epsilon)
    all_ratings_arr.append(ratings)
    all_MOS_measured.append(measuredMOS)
    #ratings = only_scoring_process(psi, delta, epsilon)


# same here ale wszystkie wartości symulacji branych pod uwagę
all_psi = np.vstack(all_psi_arr)
all_delta = np.vstack(all_delta_arr)
all_epsilon = np.vstack(all_epsilon_arr)
all_ratings = np.vstack(all_ratings_arr)
all_MOS_measured = np.dstack(all_MOS_measured)
all_MOS_measured_mean = np.mean(all_MOS_measured, axis=2)

# gathering data into matrixes and store them using compressed_pickle
algorithms = [] # lepsza nazwa by się przydała
all_rmse, all_rmse_compare, algorithms = includeParamInComparison('rmse', all_rmse, all_rmse_compare_arr, algorithms)
all_outlier, all_OutlierRatio_compare_arr, algorithms = includeParamInComparison('outlier', all_outlier, all_OutlierRatio_compare_arr, algorithms)
all_pearson, all_Pearson_compare_arr, algorithms = includeParamInComparison('pearson', all_pearson, all_pearson_compare_arr, algorithms)


algorithms_compared = compare_algorithms(number_of_metrics, algorithms)
# print("--- %s seconds ---" % (time.time() - start_time))