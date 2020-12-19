from generate_data import normal

acr = [1, 2, 3, 4, 5]


def add_noise(data, mean, st_deviation, size):
    model = data + normal(mean, st_deviation, size)
    model[model > acr[-1]] = acr[-1]
    model[model < acr[0]] = acr[0]
    return model


def load_models(psi, number_of_samples):
    models = []
    predicted_mos_best = psi
    models.append(predicted_mos_best)
    predicted_mos_good = add_noise(psi, 0, 0.25, number_of_samples)
    models.append(predicted_mos_good)
    predicted_mos_bad = add_noise(psi, 0, 0.5, number_of_samples)
    models.append(predicted_mos_bad)
    predicted_mos_worse = add_noise(psi, 0, 0.75, number_of_samples)
    models.append(predicted_mos_worse)
    predicted_mos_the_worst = add_noise(psi, 0, 1, number_of_samples)
    models.append(predicted_mos_the_worst)
    return models
