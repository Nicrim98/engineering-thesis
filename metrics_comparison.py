import numpy as np


def compare_metrics(number_of_models, metrics):
    if number_of_models > 1:
        # Podsumowanie wyników porównań wszystkich metryk w kontekście każdego parametru
        # rmse = decompress_pickle('all_rmse_compared.pbz2').astype(np.float)
        # outlier = decompress_pickle('all_outlierRatio_compared.pbz2').astype(np.float)
        # pearson = decompress_pickle('all_pearson_compared.pbz2').astype(np.float)

        stacked_metrics = np.stack(metrics, axis=2)
        number_of_simulations = stacked_metrics.shape[-1]

        metrics_summary = []
        for matrix in metrics:
            matrix_upper = np.count_nonzero(matrix == 1, axis=2) / number_of_simulations * 100
            matrix_lower = np.count_nonzero(matrix == -1, axis=2) / number_of_simulations * 100
            matrix_lower = matrix_lower.T
            metrics_summary.append(matrix_lower + matrix_upper)
        results_matrix = np.stack(metrics_summary, axis=2)
        return results_matrix
    else:
        print("WARNING ! In order to compare objective models, there needs to be at least two of them")
