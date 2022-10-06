"""
Shan, G. and Wang, W., “ExactCIdiff: an R package for computing exact confidence intervals for the difference of two proportions”, The R Journal, 5(2), 62-71 (2013).
"""

import numpy as np
from scipy.special import comb

from .._utils import accelerator  # noqa: F401


__all__ = ["_wang_lower_upper_bounds"]


# @accelerator.accelerator
def _wang_lower_upper_bounds(
    n_positive: int,
    n_total: int,
    ref_positive: int,
    ref_total: int,
    sides: int,  # -1 for left-sided, 1 for right-sided
    conf_level: float = 0.95,
    precision: float = 1e-5,
    grid: np.ndarray = [30, 20],
) -> np.ndarray:
    pround = int(np.ceil(np.log10(1 / precision)))
    datavectorL = n_positive * (ref_total + 2) + ref_positive
    datavectorU = (n_total - n_positive) * (ref_total + 2) + ref_total - ref_positive
    delta = 1e-10
    alpha = 1 - conf_level

    f_arr = np.full(((n_total + 1) * (ref_total + 1), 6), np.nan)
    S_arr = np.full(((n_total + 1) * (ref_total + 1), 2), np.nan)
    N_arr = np.full(((n_total + 1) * (ref_total + 1), 3), np.nan)
    NC_arr = np.full(((n_total + 1) * (ref_total + 1), 3), np.nan)
    Ls_arr = np.full(((n_total + 1) * (ref_total + 1), 6), np.nan)

    num = 0
    for i in range(n_total + 1):
        for j in range(ref_total + 1):
            f_arr[num, :2] = [i, j]
            num += 1
    p1_hat = f_arr[:, 0] / n_total
    p0_hat = f_arr[:, 1] / ref_total
    denom = p1_hat * (1 - p1_hat) / n_total + p0_hat * (1 - p0_hat) / ref_total + delta
    f_arr[:, 2] = (p1_hat - p0_hat) / np.sqrt(denom)
    f_arr = f_arr[f_arr[:, 2].argsort()[::-1], :]

    allvector = f_arr[:, 0] * (ref_total + 2) + f_arr[:, 1]
    allvector = np.round(allvector)
    allvectormove = (f_arr[:, 0] + 1) * (ref_total + 3) + (f_arr[:, 1] + 1)
    allvectormove = np.round(allvectormove)

    ############### for the first table ############################  # noqa: E266

    I1 = f_arr[0, 0]
    I2 = f_arr[0, 1]

    imax = 1 - 100 * delta
    imin = -1 + delta
    while abs(imax - imin) >= 1e-5:
        mid = (imax + imin) / 2
        if _wang_prob(mid, delta, n_total, ref_total, I1, I2) >= alpha:
            imax = mid
        else:
            imin = mid
    f_arr[0, 3] = np.round(imin, pround)
    Ls_arr[0, :] = f_arr[0, :].copy()

    partvector = np.round(Ls_arr[0, 0] * (ref_total + 2) + Ls_arr[0, 1])
    allvector = allvector[~np.isin(allvector, partvector)]

    ################### from the second table  ################################  # noqa: E266

    morepoint = 1
    kk = 1
    kk1 = 1
    dimoftable = Ls_arr.shape[0]

    lower, upper = -1, 1
    if n_positive == n_total and ref_positive == 0 and sides == -1:  # -1 for left-sided
        lower = Ls_arr[0, 3]
        upper = 1
        kk = dimoftable
    elif (
        n_positive == 0 and ref_positive == ref_total and sides == 1
    ):  # 1 for right-sided
        lower = -1
        upper = Ls_arr[0, 3]
        kk = dimoftable
    elif (
        n_positive == 0 and ref_positive == ref_total and sides == -1
    ):  # -1 for left-sided
        kk = dimoftable
    elif (
        n_positive == n_total and ref_positive == 0 and sides == 1
    ):  # 1 for right-sided
        kk = dimoftable

    while kk <= dimoftable - 2:
        C = Ls_arr[kk - morepoint : kk, :2].copy()
        S_arr[kk - morepoint : kk, :] = C.copy()
        DD = S_arr[:kk, :].copy()
        DD[:, 0] -= 1
        A = DD.copy()
        DD = S_arr[:kk, :].copy()
        DD[:, 1] += 1
        B = DD.copy()

        ################### generate N_arr #####  # noqa: E266
        N_arr = np.unique(np.concatenate((A, B), axis=0), axis=0)
        Nvector = (1 + N_arr[:, 0]) * (ref_total + 3) + N_arr[:, 1] + 1
        Nvector = Nvector[np.isin(Nvector, allvectormove)]

        SKvector = (1 + S_arr[:kk, 0]) * (ref_total + 3) + S_arr[:kk, 1] + 1
        Nvectortemp = Nvector[np.isin(Nvector, SKvector, invert=True)].copy()
        Ntemp = np.full((Nvectortemp.ravel().shape[0], 2), np.nan)
        Ntemp[:, 1] = np.remainder(Nvectortemp, ref_total + 3)
        Ntemp[:, 0] = (Nvectortemp - Ntemp[:, 1]) / (ref_total + 3) - 1
        Ntemp[:, 1] -= 1
        N_arr = Ntemp.copy()

        #################### generate NC_arr ####  # noqa: E266
        Nvector = (N_arr[:, 0] + 1) * (ref_total + 3) + N_arr[:, 1] + 1
        Nvector1 = (N_arr[:, 0] + 2) * (ref_total + 3) + N_arr[:, 1] + 1
        Nvector2 = (N_arr[:, 0] + 1) * (ref_total + 3) + N_arr[:, 1] + 0
        drop = np.isin(Nvector1, Nvector).astype(int) + np.isin(
            Nvector2, Nvector
        ).astype(int)
        M = np.concatenate((N_arr, np.expand_dims(drop, axis=1)), axis=1)
        MM = M[M[:, 2] < 0.5, :].copy()
        NC_arr = MM.copy()

        if NC_arr.ravel().shape[0] <= 3:
            lengthNC = 1
            NMN = np.full((2, 3), 100)
            NMN[1, :] = NC_arr.copy()
            NC_arr = NMN.copy()
        else:
            lengthNC = (1 - np.isnan(NC_arr[:, 0]).astype(int)).sum()

        for i in range(lengthNC):
            imax = 1 - 100 * delta
            imin = -1 + delta
            Ls_arr[kk, :2] = NC_arr[i, :2].copy()
            I1 = Ls_arr[: (kk + 1), 0].copy()
            I2 = Ls_arr[: (kk + 1), 1].copy()

            if kk > (dimoftable - 2) / 2:
                partvector = np.round(NC_arr[i, 0] * (ref_total + 2) + NC_arr[i, 1])
                leftvector = allvector[~np.isin(allvector, partvector)].copy()
                I2 = np.round(np.remainder(leftvector, ref_total + 2))
                I1 = np.round((leftvector - I2) / (ref_total + 2))

            imax = Ls_arr[kk - 1, 3]
            imin = -1 + delta
            if i == 0:
                NCmax = imin

            if kk <= (dimoftable - 2) / 2:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    if (
                        _wang_prob2step(mid, delta, n_total, ref_total, I1, I2, grid)
                        >= alpha
                    ):
                        imax = mid
                    else:
                        imin = mid
                NC_arr[i, 2] = np.round(imin, 2)
                if imax >= NCmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        if (
                            _wang_prob2step(
                                mid, delta, n_total, ref_total, I1, I2, grid
                            )
                            >= alpha
                        ):
                            imax = mid
                        else:
                            imin = mid
                    NC_arr[i, 2] = np.round(imin, pround)
                NCmax = max(NCmax, imin)
            else:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    if (
                        _wang_prob2steplmin(
                            mid, delta, n_total, ref_total, I1, I2, grid
                        )
                        >= 1 - alpha
                    ):
                        imin = mid
                    else:
                        imax = mid
                NC_arr[i, 2] = np.round(imin, 2)
                if imax >= NCmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        if (
                            _wang_prob2steplmin(
                                mid, delta, n_total, ref_total, I1, I2, grid
                            )
                            >= 1 - alpha
                        ):
                            imin = mid
                        else:
                            imax = mid
                    NC_arr[i, 2] = np.round(imin, pround)
                NCmax = max(NCmax, imin)

        if i >= 1:
            NCnomiss = NC_arr[
                : NC_arr[~np.isnan(NC_arr).any(axis=1)].shape[0], :
            ].copy()
            NCnomiss = NCnomiss[np.argsort(NCnomiss[:, 2]), :]
            morepoint = (NCnomiss[:, 2] >= NCnomiss[0, 2] - delta).sum()
            if morepoint >= 2:
                Ls_arr[kk : kk + morepoint, :2] = NCnomiss[:morepoint, :2].copy()
                NCres, I1, I2, imin, imax, NCmax = _wang_morepointLsest(
                    kk + morepoint,
                    kk,
                    i,
                    NCmax,
                    delta,
                    precision,
                    pround,
                    alpha,
                    n_total,
                    ref_total,
                    Ls_arr,
                    grid,
                )
                Ls_arr[kk : kk + morepoint, 3] = NCres
                for iq in range(morepoint):
                    partvector = np.round(
                        Ls_arr[kk + iq, 0] * (ref_total + 2) + Ls_arr[kk + iq, 1]
                    )
                    allvector = allvector[~np.isin(allvector, partvector)]
                kk += morepoint
            else:
                Ls_arr[kk, :2] = NCnomiss[0, :2].copy()
                Ls_arr[kk, 3] = NCnomiss[0, 2].copy()
                partvector = np.round(Ls_arr[kk, 0] * (ref_total + 2) + Ls_arr[kk, 1])
                allvector = allvector[~np.isin(allvector, partvector)]
                kk += 1
        else:
            NCnomiss = NC_arr.copy()
            Ls_arr[kk, :2] = NCnomiss[0, :2].copy()
            Ls_arr[kk, 3] = NCnomiss[0, 2].copy()
            partvector = np.round(Ls_arr[kk, 0] * (ref_total + 2) + Ls_arr[kk, 1])
            allvector = allvector[~np.isin(allvector, partvector)]
            kk += 1

        if sides == -1:
            if np.isin(datavectorL, allvector).sum() == 0:
                for jj in range(kk1, kk):
                    if Ls_arr[jj, 0] == n_positive and Ls_arr[jj, 1] == ref_positive:
                        lower = Ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    NCres, I1, I2, imin, imax, NCmax = _wang_morepointLsest(
                        kk,
                        kk,
                        i,
                        NCmax,
                        delta,
                        precision,
                        pround,
                        alpha,
                        n_total,
                        ref_total,
                        Ls_arr,
                        grid,
                    )
                    lower = NCres
                upper = 1
                kk = dimoftable
        else:
            if np.isin(datavectorU, allvector).sum() == 0:
                for jj in range(kk1, kk):
                    if Ls_arr[jj, 0] == (n_total - n_positive) and Ls_arr[jj, 1] == (
                        ref_total - ref_positive
                    ):
                        upper = Ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    NCres, I1, I2, imin, imax, NCmax = _wang_morepointLsest(
                        kk,
                        kk,
                        i,
                        NCmax,
                        delta,
                        precision,
                        pround,
                        alpha,
                        n_total,
                        ref_total,
                        Ls_arr,
                        grid,
                    )
                    upper = -NCres
                lower = -1
                kk = dimoftable

        kk1 = kk
    return np.array([lower, upper])


# @accelerator.accelerator
def _wang_prob(
    delv: float,
    delta: float,
    n_total: int,
    ref_total: int,
    I1: np.ndarray,  # or int
    I2: np.ndarray,  # or int
) -> float:
    if delv < 0:
        start = delta - delv
        stop = 1 - delta
    else:
        start = delta
        stop = 1 - delta - delv
    p0 = np.linspace(start, stop, 500)

    part1 = (
        np.atleast_2d(np.log(comb(n_total, I1))).T
        + np.atleast_2d(I1).T @ np.log(p0 + delv)[np.newaxis, :]
        + np.atleast_2d(n_total - I1).T @ np.log(1 - p0 - delv)[np.newaxis, :]
    )
    part2 = (
        np.atleast_2d(np.log(comb(ref_total, I2, exact=True))).T
        + np.atleast_2d(I2).T * np.log(p0)[np.newaxis, :]
        + np.atleast_2d(ref_total - I2).T * np.log(1 - p0)[np.newaxis, :]
    )

    return np.exp(part1 + part2).sum(axis=0).max()


# @accelerator.accelerator
def _wang_prob2step(
    delv: float,
    delta: float,
    n_total: int,
    ref_total: int,
    I1: np.ndarray,  # or int
    I2: np.ndarray,  # or int
    grid: np.array,
) -> float:
    if delv < 0:
        start = delta - delv
        stop = 1 - delta
    else:
        start = delta
        stop = 1 - delta - delv
    p0 = np.linspace(start, stop, grid[0])
    part1 = (
        np.atleast_2d(np.log(comb(n_total, I1))).T
        + np.atleast_2d(I1).T @ np.log(p0 + delv)[np.newaxis, :]
        + np.atleast_2d(n_total - I1).T @ np.log(1 - p0 - delv)[np.newaxis, :]
    )
    part2 = (
        np.atleast_2d(np.log(comb(ref_total, I2))).T
        + np.atleast_2d(I2).T * np.log(p0)[np.newaxis, :]
        + np.atleast_2d(ref_total - I2).T * np.log(1 - p0)[np.newaxis, :]
    )
    sumofprob = np.exp(part1 + part2).sum(axis=0)
    mansum = sumofprob.max()
    stepv = (p0[-1] - p0[0]) / grid[1]
    maxloc = np.where(sumofprob == mansum)[0][0]
    lowerbound = max(p0[0], p0[maxloc] - stepv) + delta
    upperbound = min(p0[-1], p0[maxloc] + stepv) - delta
    p0 = np.linspace(lowerbound, upperbound, grid[1])
    part1 = (
        np.atleast_2d(np.log(comb(n_total, I1))).T
        + np.atleast_2d(I1).T @ np.log(p0 + delv)[np.newaxis, :]
        + np.atleast_2d(n_total - I1).T @ np.log(1 - p0 - delv)[np.newaxis, :]
    )
    part2 = (
        np.atleast_2d(np.log(comb(ref_total, I2))).T
        + np.atleast_2d(I2).T * np.log(p0)[np.newaxis, :]
        + np.atleast_2d(ref_total - I2).T * np.log(1 - p0)[np.newaxis, :]
    )
    return np.exp(part1 + part2).sum(axis=0).max()


# @accelerator.accelerator
def _wang_prob2steplmin(
    delv: float,
    delta: float,
    n_total: int,
    ref_total: int,
    I1: np.ndarray,  # or int
    I2: np.ndarray,  # or int
    grid: np.array,
) -> float:
    if delv < 0:
        start = delta - delv
        stop = 1 - delta
    else:
        start = delta
        stop = 1 - delta - delv
    p0 = np.linspace(start, stop, grid[0])
    part1 = (
        np.atleast_2d(np.log(comb(n_total, I1))).T
        + np.atleast_2d(I1).T @ np.log(p0 + delv)[np.newaxis, :]
        + np.atleast_2d(n_total - I1).T @ np.log(1 - p0 - delv)[np.newaxis, :]
    )
    part2 = (
        np.atleast_2d(np.log(comb(ref_total, I2))).T
        + np.atleast_2d(I2).T * np.log(p0)[np.newaxis, :]
        + np.atleast_2d(ref_total - I2).T * np.log(1 - p0)[np.newaxis, :]
    )
    sumofprob = np.exp(part1 + part2).sum(axis=0)
    mansum = sumofprob.min()
    stepv = (p0[-1] - p0[0]) / grid[1]
    maxloc = np.where(sumofprob == mansum)[0][0]
    lowerbound = max(p0[0], p0[maxloc] - stepv) + delta
    upperbound = min(p0[-1], p0[maxloc] + stepv) - delta
    p0 = np.linspace(lowerbound, upperbound, grid[1])
    part1 = (
        np.atleast_2d(np.log(comb(n_total, I1))).T
        + np.atleast_2d(I1).T @ np.log(p0 + delv)[np.newaxis, :]
        + np.atleast_2d(n_total - I1).T @ np.log(1 - p0 - delv)[np.newaxis, :]
    )
    part2 = (
        np.atleast_2d(np.log(comb(ref_total, I2))).T
        + np.atleast_2d(I2).T * np.log(p0)[np.newaxis, :]
        + np.atleast_2d(ref_total - I2).T * np.log(1 - p0)[np.newaxis, :]
    )
    return np.exp(part1 + part2).sum(axis=0).min()


# @accelerator.accelerator
def _wang_morepointLsest(
    morekk: int,
    kk: int,
    i: int,
    NCmax: int,
    delta: float,
    precision: float,
    pround: int,
    alpha: float,
    n_total: int,
    ref_total: int,
    Ls_arr: np.ndarray,
    grid: np.ndarray,
) -> tuple:
    imax = 1 - 100 * delta
    imin = delta - 1
    I1 = Ls_arr[:morekk, 0].copy()
    I2 = Ls_arr[:morekk, 1].copy()

    imax = Ls_arr[kk - 1, 3]
    imin = -1 + delta
    if i == 0:
        NCmax = imin

    while abs(imax - imin) >= precision:
        mid = (imax + imin) / 2
        if _wang_prob2step(mid, delta, n_total, ref_total, I1, I2, grid) >= alpha:
            imax = mid
        else:
            imin = mid

    NCres = np.round(imin, pround)
    return NCres, I1, I2, imin, imax, NCmax
