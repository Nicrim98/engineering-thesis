import numpy as np
import time
from generate_data import compressed_pickle, decompress_pickle, generate_subjective_data, normal, scoring_process
from objective_models import *
import rmse as rmse
import outlier_ratio as outlier
import pearson as pearson
from metrics_comparison import compare_metrics

sim_loop = 1000
acr = [1, 2, 3, 4, 5]
all_mos_measured = []

all_psi_arr = []
all_delta_arr = []
all_epsilon_arr = []
all_ratings_arr = []

all_rmse = []
all_outlier = []
all_pearson = []

all_rmse_ci = []
all_outlier_ci = []
all_pearson_ci = []

all_rmse_compare_arr = []
all_pearson_compare_arr = []
all_OutlierRatio_compare_arr = []

params = []


def include_param_in_comparison(param_name, all_values, all_values_compare_arr, parameters_arr):
    all_values = np.vstack(all_values)
    all_values_compared = np.stack(all_values_compare_arr, axis=2)
    parameters_arr.append(all_values_compared)
    compressed_pickle('all_'+param_name+'_compared', all_values_compared)
    return all_values, all_values_compared, parameters_arr


start_time = time.time()
for number in range(0, sim_loop, 1):
    print(number)
    psi, delta, epsilon, ratings, measured_mos = generate_subjective_data(number)

    number_of_sequences = ratings.shape[0]
    number_voters = ratings.shape[1]

    # OBJECTIVE MODELS
    predicted_mos_arr = load_models(psi, number_of_sequences)
    predicted_mos = np.dstack(predicted_mos_arr)
    number_of_metrics = predicted_mos.shape[2]

    # METRICS
    rmse.get_rmse(measured_mos, predicted_mos, all_rmse, all_rmse_ci, all_rmse_compare_arr)
    outlier.get_outlier(measured_mos, predicted_mos, ratings, all_outlier, all_outlier_ci, all_OutlierRatio_compare_arr)
    pearson.get_pearson(measured_mos, predicted_mos, all_pearson, all_pearson_ci, all_pearson_compare_arr)

    # ### Liczenie R^2 dla psi lepiej działa niż peaeson correlation coefficient XD
    # r_squared = []
    # for met in range(0, number_of_metrics):
    #     mean_score_PVS_tmp = np.mean(stacked_all_metrics[0, :, met], axis=0)
    #     ones_1 = np.ones(number_of_samples)
    #     mean_score_PVS = np.outer(mean_score_PVS_tmp, ones_1)
    #     licznik = np.sum(np.subtract(MOS_measured, mean_score_PVS) * np.subtract(MOS_measured, mean_score_PVS))
    #     mianownik = np.sum(np.subtract(stacked_all_metrics[0, :, met], mean_score_PVS) * np.subtract(
    #     stacked_all_metrics[0, :, met], mean_score_PVS))
    #     wynik_psi = licznik / mianownik
    #     r_squared.append(wynik_psi)
    #
    #     r_squared_final = np.hstack(r_squared)

    # wartości zawsze występujące w mojej symulacji
    all_psi_arr.append(psi)
    all_delta_arr.append(delta)
    all_epsilon_arr.append(epsilon)
    all_ratings_arr.append(ratings)
    all_mos_measured.append(measured_mos)
    # ratings = only_scoring_process(psi, delta, epsilon)


# same here ale wszystkie wartości symulacji branych pod uwagę
all_psi = np.vstack(all_psi_arr)
all_delta = np.vstack(all_delta_arr)
all_epsilon = np.vstack(all_epsilon_arr)
all_ratings = np.stack(all_ratings_arr, axis=2)
all_MOS_measured = np.dstack(all_mos_measured)
all_MOS_measured_mean = np.mean(all_MOS_measured, axis=2)

# gathering data into matrixes and final comparison
metrics = []
all_rmse, all_rmse_compare, metrics = include_param_in_comparison('rmse', all_rmse, all_rmse_compare_arr, algorithms)
all_outlier, all_OutlierRatio_compare_arr, metrics = include_param_in_comparison(
    'outlier', all_outlier, all_OutlierRatio_compare_arr, metrics)
all_pearson, all_Pearson_compare_arr, metrics = include_param_in_comparison(
    'pearson', all_pearson, all_pearson_compare_arr, metrics)

metrics_compared = compare_metrics(number_of_metrics, metrics)
print("--- %s seconds ---" % (time.time() - start_time))
