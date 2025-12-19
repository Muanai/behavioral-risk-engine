def rolling_mean_python(values, user_ids, window):
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        c = 0
        uid = user_ids[i]
        for j in range(i, -1, -1):
            if user_ids[j] != uid:
                break
            if i - j >= window:
                break
            s += values.iloc[j]
            c += 1
        out[i] = s / c if c > 0 else 0.0
    return out
