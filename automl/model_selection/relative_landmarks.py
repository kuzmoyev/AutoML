import numpy as np


def arr(ai, ti, aj, tj, x):
    return (ai / aj) / (1 + np.log10(ti / tj) * x)


def get_arr_matrix(df, x):
    n = df.shape[0]
    arrs = np.zeros((n, n))
    for i, (_, ai, ti) in df.iterrows():
        for j, (_, aj, tj) in df.iterrows():
            if i != j:
                arrs[i][j] = arr(ai, ti, aj, tj, x)
    return arrs


def rls(arrs):
    return arrs.sum(axis=1) / (arrs.shape[0] - 1)


def get_rls(df, x):
    arrs = get_arr_matrix(df, x)
    return rls(arrs)
