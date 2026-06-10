import numpy as np
from numba import njit


# Numba-accelerated feature kernels for production use
@njit
def rolling_mean_numba(values, user_ids, window):
    n = len(values)
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        c = 0
        uid = user_ids[i]
        for j in range(i, -1, -1):
            if user_ids[j] != uid:
                break
            if i - j >= window:
                break
            s += values[j]
            c += 1
        out[i] = s / c if c > 0 else 0.0
    return out


@njit(cache=True)
# pay_matrix[i, j] > 0 indicates late payment at month j
def get_max_consecutive_late(pay_matrix):
    n_rows, n_cols = pay_matrix.shape
    result = np.zeros(n_rows, dtype=np.int32)

    for i in range(n_rows):
        max_streak = 0
        current_streak = 0
        for j in range(n_cols):
            if pay_matrix[i, j] > 0:
                current_streak += 1
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                current_streak = 0

        if current_streak > max_streak:
            max_streak = current_streak

        result[i] = max_streak
    return result


@njit(cache=True)
# Linear regression slope for x = [0,1,2] (n=3):
# slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
def get_velocity_slope(ratio_matrix):
    n_rows, n_cols = ratio_matrix.shape
    slopes = np.zeros(n_rows, dtype=np.float64)
    denom = 6.0
    for i in range(n_rows):
        sum_y = 0.0
        sum_xy = 0.0

        for x in range(n_cols):
            y = ratio_matrix[i, x]
            if np.isnan(y) or np.isinf(y):
                y = 0.0

            sum_y += y
            sum_xy += (x * y)

        num = (3.0 * sum_xy) - (3.0 * sum_y)
        slopes[i] = num / denom
    return slopes


@njit(cache=True)
def get_critical_utilization(bill_matrix, limit_array):
    n_rows, n_cols = bill_matrix.shape
    counts = np.zeros(n_rows, dtype=np.int32)

    for i in range(n_rows):
        limit = limit_array[i]
        if limit == 0: continue

        c = 0
        threshold = limit * 0.9

        for j in range(n_cols):
            if bill_matrix[i, j] > threshold:
                c += 1
        counts[i] = c
    return counts
