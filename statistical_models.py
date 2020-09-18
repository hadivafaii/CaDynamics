import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.metrics import r2_score


def mar(data, time_lags=2, xv_folds=10, which_fold=None, normalize=True, verbose=True):
    nt, nc = data.shape
    m = nt - time_lags
    num_params = time_lags * nc ** 2 + (nc ** 2 + nc) / 2

    msg = "nt: {:d}, nc: {:d}, num lags: {:d},\nnum usable samples: {:d}, num params: {:d}"
    msg = msg.format(nt, nc, time_lags, m, int(num_params))
    if verbose:
        print(msg)

    if normalize:
        x = zscore(data)
    else:
        x = data

    c_list = []
    for t in range(m):
        c_list.append(x[range(t, t + time_lags)].reshape(1, -1))

    c = np.concatenate(c_list)
    y = x[time_lags:]

    if which_fold is None:
        which_fold = xv_folds // 2
    tst_ids = list(range(which_fold * m // xv_folds, (which_fold + 1) * m // xv_folds))
    trn_ids = list(set(np.arange(m)).difference(set(tst_ids)))
    assert not set(trn_ids).intersection(set(tst_ids)), "train/test indices must be disjoint"

    c_trn, c_tst = c[trn_ids], c[tst_ids]
    y_trn, y_tst = y[trn_ids], y[tst_ids]

    a_ml = np.linalg.solve(c_trn.T @ c_trn, c_trn.T @ y_trn)
    delta = y_trn - c_trn @ a_ml
    sigma = delta.T @ delta
    sigma_ml = np.diag(sigma) / m

    y_pred_trn = c_trn @ a_ml
    y_pred_tst = c_tst @ a_ml
    r2_trn = r2_score(y_trn, y_pred_trn, multioutput='raw_values') * 100
    r2_tst = r2_score(y_tst, y_pred_tst, multioutput='raw_values') * 100

    outputs = {
        "a_ml": a_ml,
        "sigma_ml": sigma_ml,
        "r2_trn": r2_trn,
        "r2_tst": r2_tst,
        "c": c,
        "y": y,
        "y_tst": y_tst,
        "y_pred_tst": y_pred_tst,
    }

    return outputs


def run_mar_analysis(data: np.ndarray, label: str, max_lags: int, xv_folds: int = 10, normalize: bool = True):
    mar_results = pd.DataFrame(columns=["label", "lags", "fold", "r2"])

    for k in tqdm(range(1, max_lags)):
        for f in range(xv_folds):
            outputs_dict = mar(
                data, time_lags=k,
                xv_folds=xv_folds, which_fold=f,
                normalize=normalize, verbose=False)
            r2 = outputs_dict["r2_tst"]
            r2_plus = np.maximum(r2, 0.0)
            r2_mean = np.round(np.mean(r2_plus), decimals=3)

            df = pd.DataFrame(
                data={
                    "label": [label],
                    "lags": [k], "fold": [f],
                    "r2": [r2_mean],
                })
            mar_results = mar_results.append(df)

    return mar_results.reset_index()
