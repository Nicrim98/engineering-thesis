import numpy as np
import time
from generate_data import generate_subjective_data
from objective_models import *
import rmse as rmse
import outlier_ratio as outlier
import pearson as pearson
from metrics_comparison import compare_metrics
import _pickle as cpickle
import bz2

sim_loop = 1000
acr = [1, 2, 3, 4, 5]

# input data
all_psi_arr = []
all_delta_arr = []
all_epsilon_arr = []
all_ratings_arr = []
all_mos_measured = []
all_mos_predicted = []

# data under evaluation
all_rmse = []
all_outlier = []
all_pearson = []
all_rmse_ci = []
all_outlier_ci = []
all_pearson_ci = []

# output data
all_rmse_compared_arr = []
all_pearson_compared_arr = []
all_outlier_ratio_compared_arr = []

params = []

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cpickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cpickle.load(data)
    return data


def include_param_in_comparison(param_name, all_values, all_values_ci, all_values_compare_arr, parameters_arr):
    all_values = np.vstack(all_values)
    all_values_ci = np.stack(all_values_ci)
    all_values_compared = np.stack(all_values_compare_arr, axis=2)
    parameters_arr.append(all_values_compared)
    compressed_pickle('all_'+param_name+'_compared', all_values_compared)
    return all_values, all_values_ci, all_values_compared, parameters_arr


start_time = time.time()
for sim_iteration in range(0, sim_loop, 1):
    psi, delta, epsilon, ratings, measured_mos = generate_subjective_data()

    number_of_sequences = ratings.shape[0]
    number_voters = ratings.shape[1]

    # OBJECTIVE MODELS
    predicted_mos_arr = load_models(psi, number_of_sequences)
    predicted_mos = np.dstack(predicted_mos_arr)
    number_of_models = predicted_mos.shape[2]

    # METRICS
    rmse.get_rmse(measured_mos, predicted_mos, all_rmse, all_rmse_ci, all_rmse_compared_arr)
    outlier.get_outlier(measured_mos, predicted_mos, ratings, all_outlier, all_outlier_ci, all_outlier_ratio_compared_arr)
    pearson.get_pearson(measured_mos, predicted_mos, all_pearson, all_pearson_ci, all_pearson_compared_arr)

    # stacking values related to the simulation for archiving purposes
    all_psi_arr.append(psi)
    all_delta_arr.append(delta)
    all_epsilon_arr.append(epsilon)
    all_ratings_arr.append(ratings)
    all_mos_measured.append(measured_mos)
    all_mos_predicted.append(predicted_mos)


all_psi = np.vstack(all_psi_arr)
all_delta = np.vstack(all_delta_arr)
all_epsilon = np.vstack(all_epsilon_arr)
all_ratings = np.stack(all_ratings_arr, axis=2)
all_mos_measured = np.dstack(all_mos_measured)
all_mos_measured_mean = np.mean(all_mos_measured, axis=2)

# gathering data into matrixes and final comparison
metrics = []
all_rmse, all_rmse_ci, all_rmse_compared, metrics = include_param_in_comparison(
    'rmse', all_rmse, all_rmse_ci, all_rmse_compared_arr, metrics)
all_outlier, all_outlier_ci, all_outlier_ratio_compared, metrics = include_param_in_comparison(
    'outlier', all_outlier, all_outlier_ci, all_outlier_ratio_compared_arr, metrics)
all_pearson, all_pearson_ci, all_pearson_compared, metrics = include_param_in_comparison(
    'pearson', all_pearson, all_pearson_ci, all_pearson_compared_arr, metrics)

metrics_compared = compare_metrics(number_of_models, metrics)

# compressed_pickle('output_data/scenario_3/psi', all_psi)
# compressed_pickle('output_data/scenario_3/delta', all_delta)
# compressed_pickle('output_data/scenario_3/epsilon', all_epsilon)
# compressed_pickle('output_data/scenario_3/ratings', all_ratings)
# compressed_pickle('output_data/scenario_3/mos_o', measured_mos)
# compressed_pickle('output_data/scenario_3/mos_p', predicted_mos)
# compressed_pickle('output_data/scenario_3/rmse', all_rmse)
# compressed_pickle('output_data/scenario_3/rmse_ci', all_rmse_ci)
# compressed_pickle('output_data/scenario_3/outlier_ratio', all_outlier)
# compressed_pickle('output_data/scenario_3/outlier_ratio_ci', all_outlier_ci)
# compressed_pickle('output_data/scenario_3/pearson', all_pearson)
# compressed_pickle('output_data/scenario_3/pearson_ci', all_pearson_ci)
# compressed_pickle('output_data/scenario_3/rmse_compared', all_rmse_compared)
# compressed_pickle('output_data/scenario_3/outlier_ratio_compared', all_outlier_ratio_compared)
# compressed_pickle('output_data/scenario_3/pearson_compared', all_pearson_compared)
# compressed_pickle('output_data/scenario_3/metrics_compared', metrics_compared)

print("--- %s seconds ---" % (time.time() - start_time))
