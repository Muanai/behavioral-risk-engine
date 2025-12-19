import numpy as np


def calculate_psi(expected, actual, bucket_type='bins', buckets=10, axis=0):
    def psi(expected_array, actual_array, buckets):
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if bucket_type == 'bins':
            breakpoints = np.percentile(expected_array, breakpoints)

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    psi_values = psi(expected, actual, buckets)
    return psi_values


def check_model_stability(model, X_train, X_test):
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    psi_score = calculate_psi(train_probs, test_probs, buckets=10)

    status = "UNKNOWN"
    if psi_score < 0.1:
        status = "GREEN (Stable)"
    elif psi_score < 0.25:
        status = "YELLOW (Minor Drift)"
    else:
        status = "RED (Major Drift - RETRAIN NEEDED!)"

    return psi_score, status
