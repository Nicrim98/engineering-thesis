import numpy as np


def compare_metrics(number_of_metrics, algorithms):
    if number_of_metrics > 1:
        # Podsumowanie wyników porównań wszystkich metryk w kontekście każdego parametru
        # rmse = decompress_pickle('all_rmse_compared.pbz2').astype(np.float)
        # outlier = decompress_pickle('all_outlierRatio_compared.pbz2').astype(np.float)
        # pearson = decompress_pickle('all_pearson_compared.pbz2').astype(np.float)

        stacked_algorithms = np.stack(algorithms, axis=2)

        number_of_simulations = stacked_algorithms.shape[-1]
        # params = [rmse, outlier, pearson]
        # stacked_parameters = np.stack(params, axis=3)

        metrics_summary = []
        for matrix in algorithms:
            matrix_upper = np.count_nonzero(matrix == 1, axis=2) / number_of_simulations * 100
            matrix_lower = np.count_nonzero(matrix == -1, axis=2) / number_of_simulations * 100
            matrix_lower = matrix_lower.T
            metrics_summary.append(matrix_lower + matrix_upper)
        results_matrix = np.stack(metrics_summary, axis=2)
        return results_matrix
    else:
        print("WARNING ! In order to compare objective models, there needs to be at least two of them")
