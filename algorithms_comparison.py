import numpy as np


def compare_algorithms(number_of_metrics, algorithms):
    if number_of_metrics > 1:
        # Podsumowanie wyników porównań wszystkich metryk w kontekście każdego parametru
        # rmse = decompress_pickle('all_rmse_compared.pbz2').astype(np.float)
        # outlier = decompress_pickle('all_outlierRatio_compared.pbz2').astype(np.float)
        # pearson = decompress_pickle('all_pearson_compared.pbz2').astype(np.float)

        stacked_algorithms = np.stack(algorithms, axis=2)

        number_of_simulations = stacked_algorithms.shape[-1]
        # params = [rmse, outlier, pearson]
        # stacked_parameters = np.stack(params, axis=3)

        algorithms_comparing = []
        for param in algorithms:
            param_left = np.count_nonzero(param == 1, axis=2) / number_of_simulations * 100  # left model is better
            param_up = np.count_nonzero(param == -1, axis=2) / number_of_simulations * 100  # up model is better
            param_up = param_up.T
            algorithms_comparing.append(param_left + param_up)
        algorithms_compared = np.stack(algorithms_comparing, axis=2)
        return algorithms_compared
    else:
        print("WARNING ! In order to compare objective models, there needs to be at least two of them")
