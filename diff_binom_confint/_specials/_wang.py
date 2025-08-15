"""
Shan, G. and Wang, W., “ExactCIdiff: an R package for computing exact confidence intervals for the difference of two proportions”, The R Journal, 5(2), 62-71 (2013).
"""

import warnings
from typing import List, Optional, Union

import numpy as np
from scipy.special import comb

from .._confint import _SIDE_NAME_MAP, ConfidenceInterval, ConfidenceIntervalSides

__all__ = ["wang_binomial_ci"]


def wang_binomial_ci(
    n_positive: int,
    n_total: int,
    ref_positive: int,
    ref_total: int,
    conf_level: float = 0.95,
    sides: Union[str, int] = "two-sided",
    precision: float = 1e-5,
    grid_one: int = 30,
    grid_two: int = 20,
    verbose: bool = False,
) -> ConfidenceInterval:
    """Calculate exact confidence intervals for the difference of two proportions
    using the method from Wang (2013).

    Parameters
    ----------
    n_positive : int
        number of positive samples.
    n_total : int
        total number of samples.
    ref_positive : int
        number of positive samples of the reference.
    ref_total : int
        total number of samples of the reference.
    conf_level : float, optional
        Confidence level, by default 0.95
    sides : Union[str, int], optional
        sides: str or int, default "two-sided",
        the sides of the confidence interval, should be one of
        "two-sided" (aliases "2-sided", "two_sided", "2_sided", "2-sides", "two_sides", "two-sides", "2_sides", "ts", "t", "two", "2", 2),
        "left-sided" (aliases "left_sided", "left", "ls", "l"),
        "right-sided" (aliases "right_sided", "right", "rs", "r"),
        case insensitive.
    precision : float, optional
        Precision for the search algorithm, by default 1e-5
    grid_one : int, optional
        Number of grid points in first step, by default 30
    grid_two : int, optional
        Number of grid points in second step, by default 20
    verbose : bool, optional
        Verbosity for debug message.

    Returns
    -------
    confint : ConfidenceInterval
        The confidence interval.
    """
    # Input validation
    if not isinstance(n_total, int) or n_total < 1:
        raise ValueError("Number of subjects n_total must be a positive integer.")
    if not isinstance(ref_total, int) or ref_total < 1:
        raise ValueError("Number of subjects ref_total must be a positive integer.")
    if not isinstance(grid_one, int) or grid_one < 1:
        raise ValueError("grid_one must be a positive integer.")
    if not isinstance(grid_two, int) or grid_two < 1:
        raise ValueError("grid_two must be a positive integer.")
    if n_total + ref_total > 100:
        warnings.warn(
            "It may take more time to compute the confidence limits.",
            RuntimeWarning,
        )
    if not isinstance(n_positive, int) or n_positive < 0 or n_positive > n_total:
        raise ValueError("n_positive must be an integer between 0 and n_total.")
    if not isinstance(ref_positive, int) or ref_positive < 0 or ref_positive > ref_total:
        raise ValueError("ref_positive must be an integer between 0 and ref_total.")
    if not (0 < conf_level < 1):
        raise ValueError("conf_level must be in (0,1).")
    if not (precision > 0):
        raise ValueError("precision must be positive.")

    sides_key = str(sides).lower()
    if sides_key not in _SIDE_NAME_MAP:
        raise ValueError(f"sides should be one of {list(_SIDE_NAME_MAP)}, but got {sides}")
    sides_val = _SIDE_NAME_MAP[sides_key]

    if sides_val == ConfidenceIntervalSides.TwoSided.value:
        adjusted = 1 - (1 - conf_level) / 2
        ci_l = binomial_ci_one_sided(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            adjusted,
            ConfidenceIntervalSides.LeftSided.value,
            precision,
            grid_one,
            grid_two,
        )
        if verbose:
            print(f"Left CI: {ci_l}")
        ci_u = binomial_ci_one_sided(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            adjusted,
            ConfidenceIntervalSides.RightSided.value,
            precision,
            grid_one,
            grid_two,
        )
        if verbose:
            print(f"Right CI: {ci_u}")
        lower, upper = ci_l[1], ci_u[2]
        estimate = ci_u[0]
        return ConfidenceInterval(lower, upper, estimate, conf_level, "wang", sides_val)
    else:
        ci = binomial_ci_one_sided(
            n_positive, n_total, ref_positive, ref_total, conf_level, sides_val, precision, grid_one, grid_two
        )
        return ConfidenceInterval(ci[1], ci[2], ci[0], conf_level, "wang", sides_val)


def binomial_ci_one_sided(
    n_positive: int,
    n_total: int,
    ref_positive: int,
    ref_total: int,
    conf_level: float,
    sides: str,
    precision: float,
    grid_one: int,
    grid_two: int,
) -> List[float]:
    """Helper function that calculates one-sided confidence interval.

    Parameters
    ----------
    n_positive : int
        number of positive samples.
    n_total : int
        total number of samples.
    ref_positive : int
        number of positive samples of the reference.
    ref_total : int
        total number of samples of the reference.
    conf_level : float, optional
        Confidence level, by default 0.95
    sides : Union[str, int], optional
        sides: str or int, default "two-sided",
        the sides of the confidence interval, should be one of
        "two-sided" (aliases "2-sided", "two_sided", "2_sided", "2-sides", "two_sides", "two-sides", "2_sides", "ts", "t", "two", "2", 2),
        "left-sided" (aliases "left_sided", "left", "ls", "l"),
        "right-sided" (aliases "right_sided", "right", "rs", "r"),
        case insensitive.
    precision : float, optional
        Precision for the search algorithm, by default 1e-5
    grid_one : int, optional
        Number of grid points in first step, by default 30
    grid_two : int, optional
        Number of grid points in second step, by default 20

    Returns
    -------
    List[float]
        A list containing the point estimate, lower bound, and upper bound of the confidence interval.

    """
    n = n_total
    m = ref_total
    x = n_positive
    y = ref_positive

    pround = 0
    while 1 / precision >= 10**pround:
        pround += 1

    datavector_l = x * (m + 2) + y
    datavector_u = (n - x) * (m + 2) + (m - y)
    output = [0.0, 0.0, 0.0]
    output[0] = round(x / n - y / m, 6)
    delta = 1e-10
    alpha = 1 - conf_level

    total_pts = (n + 1) * (m + 1)
    f = np.full((total_pts, 6), np.nan)

    # enumerate (i,j)
    idx = 0
    for i in range(n + 1):
        for j in range(m + 1):
            f[idx, 0:2] = (i, j)
            idx += 1

    p1hat = f[:, 0] / n
    p0hat = f[:, 1] / m
    denom = p1hat * (1 - p1hat) / n + p0hat * (1 - p0hat) / m + delta
    f[:, 2] = (p1hat - p0hat) / np.sqrt(denom)

    # Sort f by the third column in descending order
    f = f[(-f[:, 2]).argsort(), :]

    allvector = np.round(f[:, 0] * (m + 2) + f[:, 1]).astype(int)
    allvectormove = np.round((f[:, 0] + 1) * (m + 3) + (f[:, 1] + 1)).astype(int)

    # first table
    i1 = int(f[0, 0])
    i2 = int(f[0, 1])
    imax = 1 - 100 * delta
    imin = -1 + delta
    while abs(imax - imin) >= 1e-5:
        mid = (imax + imin) / 2
        probmid = _prob(mid, delta, n, m, np.array([i1]), np.array([i2]))
        if probmid >= alpha:
            imax = mid
        else:
            imin = mid

    f[0, 3] = round(imin, pround)

    ls_arr = np.full((total_pts, 6), np.nan)
    ls_arr[0, :] = f[0, :]

    partvector = int(round(ls_arr[0, 0] * (m + 2) + ls_arr[0, 1]))
    allvector = allvector[allvector != partvector]

    # From the second table
    morepoint = 1
    kk = 1
    kk1 = 1
    dimoftable = total_pts

    # handle special cases
    if x == n and y == 0 and sides == ConfidenceIntervalSides.LeftSided.value:
        output[1] = ls_arr[0, 3]
        output[2] = 1
        kk = dimoftable
    if x == 0 and y == m and sides == ConfidenceIntervalSides.LeftSided.value:
        output[1] = -1
        output[2] = 1
        kk = dimoftable
    if x == n and y == 0 and sides == ConfidenceIntervalSides.RightSided.value:
        output[1] = -1
        output[2] = 1
        kk = dimoftable
    if x == 0 and y == m and sides == ConfidenceIntervalSides.RightSided.value:
        output[1] = -1
        output[2] = -ls_arr[0, 3]
        kk = dimoftable

    s = np.full((total_pts, 2), np.nan)

    while kk <= dimoftable - 2:
        c = ls_arr[(kk - morepoint) : kk, 0:2].copy()
        s[(kk - morepoint) : kk, :] = c
        dd = s[:kk, :].copy()
        dd[:, 0] -= 1
        a = dd
        dd = s[:kk, :].copy()
        dd[:, 1] += 1
        b = dd

        # Generate N
        n_arr = np.unique(np.vstack((a, b)), axis=0)
        nvector = ((n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1] + 1).astype(int)
        nvector = nvector[np.isin(nvector, allvectormove)]

        skvector = ((s[:kk, 0] + 1) * (m + 3) + s[:kk, 1] + 1).astype(int)
        nvectortemp = nvector[~np.isin(nvector, skvector)]

        if len(nvectortemp) == 0:
            ntemp = np.empty((0, 2))
        else:
            ntemp = np.zeros((len(nvectortemp), 2))
            ntemp[:, 1] = np.mod(nvectortemp, m + 3)
            ntemp[:, 0] = (nvectortemp - ntemp[:, 1]) / (m + 3) - 1
            ntemp[:, 1] -= 1
        n_arr = ntemp

        # Generate NC
        if n_arr.shape[0] == 0:
            nc_arr = np.empty((0, 3))
        else:
            nvector = ((n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1] + 1).astype(int)
            nvector1 = ((n_arr[:, 0] + 2) * (m + 3) + n_arr[:, 1] + 1).astype(int)
            nvector2 = ((n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1]).astype(int)
            drop = np.isin(nvector1, nvector) * 1 + np.isin(nvector2, nvector) * 1
            m_arr = np.column_stack((n_arr, drop))
            nc_arr = m_arr[m_arr[:, 2] < 0.5, :]

        if nc_arr.size <= 3:
            length_nc = 1
            sentinel = np.full((2, 3), 100.0)
            if nc_arr.size == 3:
                sentinel[0, :] = nc_arr[0]
            nc_arr = sentinel
        else:
            length_nc = nc_arr.shape[0]

        for ci in range(length_nc):
            ls_arr[kk, 0:2] = nc_arr[ci, 0:2]
            i1_vec = ls_arr[: (kk + 1), 0]
            i2_vec = ls_arr[: (kk + 1), 1]

            if kk > (dimoftable - 2) / 2:
                partvector = int(round(nc_arr[ci, 0] * (m + 2) + nc_arr[ci, 1]))
                leftvector = allvector[allvector != partvector]
                if leftvector.size > 0:
                    i2_vec = leftvector % (m + 2)
                    i1_vec = (leftvector - i2_vec) / (m + 2)

            base_limit = ls_arr[kk - 1, 3] if kk - 1 >= 0 and not np.isnan(ls_arr[kk - 1, 3]) else (1 - 100 * delta)
            imax = base_limit
            imin = -1 + delta
            if ci == 0:
                ncmax = imin

            if kk <= (dimoftable - 2) / 2:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    probmid = _prob2step(mid, delta, n, m, i1_vec, i2_vec, grid_one, grid_two)
                    if probmid >= alpha:
                        imax = mid
                    else:
                        imin = mid
                nc_arr[ci, 2] = round(imin, 2)
                if imax >= ncmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        probmid = _prob2step(mid, delta, n, m, i1_vec, i2_vec, grid_one, grid_two)
                        if probmid >= alpha:
                            imax = mid
                        else:
                            imin = mid
                    nc_arr[ci, 2] = round(imin, pround)
                ncmax = max(ncmax, imin)
            else:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    probmid = _prob2steplmin(mid, delta, n, m, i1_vec, i2_vec, grid_one, grid_two)
                    if probmid >= 1 - alpha:
                        imin = mid
                    else:
                        imax = mid
                nc_arr[ci, 2] = round(imin, 2)
                if imax >= ncmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        probmid = _prob2steplmin(mid, delta, n, m, i1_vec, i2_vec, grid_one, grid_two)
                        if probmid >= 1 - alpha:
                            imin = mid
                        else:
                            imax = mid
                    nc_arr[ci, 2] = round(imin, pround)
                ncmax = max(ncmax, imin)

        if length_nc >= 2:
            valid = ~np.isnan(nc_arr[:, 0])
            ncnomiss = nc_arr[valid]
            ncnomiss = ncnomiss[(-ncnomiss[:, 2]).argsort(), :]
            morepoint = np.sum(ncnomiss[:, 2] >= ncnomiss[0, 2] - delta)
            if morepoint >= 2:
                ls_arr[kk : kk + morepoint, 0:2] = ncnomiss[:morepoint, 0:2]
                prev_limit = ls_arr[kk - 1, 3] if kk - 1 >= 0 else (1 - 100 * delta)
                ncres = _morepointlsest(
                    kk + morepoint,
                    delta,
                    precision,
                    pround,
                    alpha,
                    n,
                    m,
                    ls_arr,
                    grid_one,
                    grid_two,
                    prev_limit=prev_limit,
                )
                ls_arr[kk : kk + morepoint, 3] = ncres
                for iq in range(morepoint):
                    pv = int(round(ls_arr[kk + iq, 0] * (m + 2) + ls_arr[kk + iq, 1]))
                    allvector = allvector[allvector != pv]
                kk += morepoint
            else:
                ls_arr[kk, 0:2] = ncnomiss[0, 0:2]
                ls_arr[kk, 3] = ncnomiss[0, 2]
                pv = int(round(ls_arr[kk, 0] * (m + 2) + ls_arr[kk, 1]))
                allvector = allvector[allvector != pv]
                kk += 1
        else:
            # length_nc ==1
            ls_arr[kk, 0:2] = nc_arr[0, 0:2]
            ls_arr[kk, 3] = nc_arr[0, 2]
            pv = int(round(ls_arr[kk, 0] * (m + 2) + ls_arr[kk, 1]))
            allvector = allvector[allvector != pv]
            kk += 1

        if sides == ConfidenceIntervalSides.LeftSided.value:
            if not np.isin(datavector_l, allvector):
                for jj in range(kk1, kk):
                    if int(ls_arr[jj, 0]) == x and int(ls_arr[jj, 1]) == y:
                        output[1] = ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    prev_limit = ls_arr[kk - 1, 3] if kk - 1 >= 0 else (1 - 100 * delta)
                    ncres = _morepointlsest(
                        kk, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two, prev_limit=prev_limit
                    )
                    output[1] = ncres
                output[2] = 1
                kk = dimoftable
        else:
            if not np.isin(datavector_u, allvector):
                for jj in range(kk1, kk):
                    if int(ls_arr[jj, 0]) == (n - x) and int(ls_arr[jj, 1]) == (m - y):
                        output[2] = -ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    prev_limit = ls_arr[kk - 1, 3] if kk - 1 >= 0 else (1 - 100 * delta)
                    ncres = _morepointlsest(
                        kk, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two, prev_limit=prev_limit
                    )
                    output[2] = -ncres
                output[1] = -1
                kk = dimoftable

        kk1 = kk

    output = [val.item() if isinstance(val, np.generic) else val for val in output]

    return output


def _prob(delv, delta, n, m, i1, i2):
    """Calculate probability for binary search"""
    if delv < 0:
        p0 = np.linspace(-delv + delta, 1 - delta, 500)
    else:
        p0 = np.linspace(delta, 1 - delv - delta, 500)

    # Convert i1/i2 to arrays if they're not already
    i1 = np.atleast_1d(i1)
    i2 = np.atleast_1d(i2)
    part1 = np.log(comb(n, i1))[:, None] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))
    part2 = np.log(comb(m, i2))[:, None] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    return np.max(np.exp(part1 + part2).sum(axis=0))


def _prob2step(delv, delta, n, m, i1, i2, grid_one, grid_two):
    """Two-step probability calculation with grid refinement"""
    if delv < 0:
        p0 = np.linspace(-delv + delta, 1 - delta, grid_one)
    else:
        p0 = np.linspace(delta, 1 - delv - delta, grid_one)
    i1 = np.atleast_1d(i1)
    i2 = np.atleast_1d(i2)
    part1 = np.log(comb(n, i1))[:, None] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))
    part2 = np.log(comb(m, i2))[:, None] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))
    sumofprob = np.exp(part1 + part2).sum(axis=0)

    # plateau-aware refinement (R: which(sumofprob == max(sumofprob)))
    mansum = sumofprob.max()
    atol = 1e-14 * (mansum if mansum > 0 else 1.0)
    plateau_idx = np.where(np.isclose(sumofprob, mansum, rtol=0.0, atol=atol))[0]
    leftmost = plateau_idx.min()
    rightmost = plateau_idx.max()

    stepv = (p0[-1] - p0[0]) / grid_one
    lowerb = max(p0[0], p0[rightmost] - stepv) + delta
    upperb = min(p0[-1], p0[leftmost] + stepv) - delta

    # stepv = (p0[-1] - p0[0]) / grid_one
    # maxloc = np.argmax(sumofprob)
    # lowerb = max(p0[0], p0[maxloc] - stepv) + delta
    # upperb = min(p0[-1], p0[maxloc] + stepv) - delta

    p0 = np.linspace(lowerb, upperb, grid_two)
    part1 = np.log(comb(n, i1))[:, None] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))
    part2 = np.log(comb(m, i2))[:, None] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))
    return np.exp(part1 + part2).sum(axis=0).max()


def _prob2steplmin(delv, delta, n, m, i1, i2, grid_one, grid_two):
    """Two-step minimum probability calculation with grid refinement"""
    if delv < 0:
        p0 = np.linspace(-delv + delta, 1 - delta, grid_one)
    else:
        p0 = np.linspace(delta, 1 - delv - delta, grid_one)
    i1 = np.atleast_1d(i1)
    i2 = np.atleast_1d(i2)
    part1 = np.log(comb(n, i1))[:, None] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))
    part2 = np.log(comb(m, i2))[:, None] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))
    sumofprob = np.exp(part1 + part2).sum(axis=0)

    # plateau-aware refinement for minima (R: which(sumofprob == min(sumofprob)))
    mansum = sumofprob.min()
    atol = 1e-14 * (abs(mansum) if mansum != 0 else 1.0)
    plateau_idx = np.where(np.isclose(sumofprob, mansum, rtol=0.0, atol=atol))[0]
    leftmost = plateau_idx.min()
    rightmost = plateau_idx.max()

    stepv = (p0[-1] - p0[0]) / grid_one
    lowerb = max(p0[0], p0[rightmost] - stepv) + delta
    upperb = min(p0[-1], p0[leftmost] + stepv) - delta

    # stepv = (p0[-1] - p0[0]) / grid_one
    # minloc = np.argmin(sumofprob)
    # lowerb = max(p0[0], p0[minloc] - stepv) + delta
    # upperb = min(p0[-1], p0[minloc] + stepv) - delta

    p0 = np.linspace(lowerb, upperb, grid_two)
    part1 = np.log(comb(n, i1))[:, None] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))
    part2 = np.log(comb(m, i2))[:, None] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))
    return np.exp(part1 + part2).sum(axis=0).min()


def _morepointlsest(
    morekk: int,
    delta: float,
    precision: float,
    pround: int,
    alpha: float,
    n: int,
    m: int,
    ls_arr: np.ndarray,
    grid_one: int,
    grid_two: int,
    prev_limit: Optional[float] = None,
):
    """Calculate confidence interval bounds with multiple points"""
    i1 = ls_arr[:morekk, 0]
    i2 = ls_arr[:morekk, 1]
    imin = -1 + delta

    if prev_limit is None or np.isnan(prev_limit):
        imax = 1 - 100 * delta
    else:
        imax = prev_limit

    while abs(imax - imin) >= precision:
        mid = (imax + imin) / 2
        probmid = _prob2step(mid, delta, n, m, i1, i2, grid_one, grid_two)
        if probmid >= alpha:
            imax = mid
        else:
            imin = mid

    return round(imin, pround)
