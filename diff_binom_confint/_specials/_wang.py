"""
Shan, G. and Wang, W., “ExactCIdiff: an R package for computing exact confidence intervals for the difference of two proportions”, The R Journal, 5(2), 62-71 (2013).
"""

import warnings
from typing import List, Union

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
        raise ValueError("Number of grid in the first step search grid_one must be a positive integer.")
    if not isinstance(grid_two, int) or grid_two < 1:
        raise ValueError("Number of grid in the second step search grid_two must be a positive integer.")
    if n_total + ref_total > 100:
        warnings.warn(
            "It may take more time to compute the confidence limits, and the results might be incorrect.", RuntimeWarning
        )
    if not isinstance(n_positive, int) or n_positive < 0 or n_positive > n_total:
        raise ValueError("Observed number of response n_positive must be an integer between 0 and n_total.")
    if not isinstance(ref_positive, int) or ref_positive < 0 or ref_positive > ref_total:
        raise ValueError("Observed number of response ref_positive must be an integer between 0 and ref_total.")
    if not isinstance(conf_level, (int, float)) or conf_level <= 0 or conf_level >= 1:
        raise ValueError("conf_level must be a positive number between 0 and 1, default 0.95.")
    if not isinstance(precision, (int, float)) or precision <= 0:
        raise ValueError("precision must be a positive number, default 0.00001.")

    sides = str(sides).lower()
    if sides not in _SIDE_NAME_MAP:
        raise ValueError(f"sides should be one of {list(_SIDE_NAME_MAP)}, but got {sides}")
    else:
        sides = _SIDE_NAME_MAP[sides]

    if sides == ConfidenceIntervalSides.TwoSided.value:
        # For two-sided CI, calculate lower and upper bounds separately with adjusted confidence level
        adjusted_conf_level = 1 - (1 - conf_level) / 2
        ci_lower = binomial_ci_one(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            adjusted_conf_level,
            ConfidenceIntervalSides.LeftSided.value,
            precision,
            grid_one,
            grid_two,
        )
        ci_upper = binomial_ci_one(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            adjusted_conf_level,
            ConfidenceIntervalSides.RightSided.value,
            precision,
            grid_one,
            grid_two,
        )
        ci_output = [ci_lower[1], ci_upper[2]]
        confint = ConfidenceInterval(
            ci_output[0],  # lower bound
            ci_output[1],  # upper bound
            ci_upper[0],  # estimate
            conf_level,  # level
            "wang",  # method
            sides,  # sides
        )
        return confint
    else:
        ci = binomial_ci_one(n_total, ref_total, n_positive, ref_positive, conf_level, sides, precision, grid_one, grid_two)
        # result = {
        #     "conf_level": conf_level,
        #     "sides": sides,
        #     "estimate": ci[0],
        #     "exact_ci": [ci[1], ci[2]]
        # }
        confint = ConfidenceInterval(
            ci[1],  # lower bound
            ci[2],  # upper bound
            ci[0],  # estimate
            conf_level,  # level
            "wang",  # method
            sides,  # sides
        )
        return confint


def binomial_ci_one(
    n_positive, n_total, ref_positive, ref_total, conf_level, sides, precision, grid_one, grid_two
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
    datavector_u = (n - x) * (m + 2) + m - y
    output = [0, 0, 0]
    output[0] = round(x / n - y / m, 6)
    delta = 1e-10
    alpha = 1 - conf_level

    # Initialize arrays
    f = np.full(((n + 1) * (m + 1), 6), np.nan)
    s = np.full(((n + 1) * (m + 1), 2), np.nan)
    n_arr = np.full(((n + 1) * (m + 1), 3), np.nan)
    nc_arr = np.full(((n + 1) * (m + 1), 3), np.nan)
    ls_arr = np.full(((n + 1) * (m + 1), 6), np.nan)

    # Fill in initial f array
    num = 0
    for i in range(n + 1):
        for j in range(m + 1):
            f[num, 0:2] = [i, j]
            num += 1

    p1hat = f[:, 0] / n
    p0hat = f[:, 1] / m
    denom = p1hat * (1 - p1hat) / n + p0hat * (1 - p0hat) / m + delta
    f[:, 2] = (p1hat - p0hat) / np.sqrt(denom)

    # Sort f by the third column in descending order
    f = f[(-f[:, 2]).argsort(), :]

    allvector = f[:, 0] * (m + 2) + f[:, 1]
    allvector = np.round(allvector)

    allvectormove = (f[:, 0] + 1) * (m + 3) + (f[:, 1] + 1)
    allvectormove = np.round(allvectormove)

    # For the first table
    i1 = f[0, 0]
    i2 = f[0, 1]

    imax = 1 - 100 * delta
    imin = -1 + delta
    while abs(imax - imin) >= 1e-5:
        mid = (imax + imin) / 2
        probmid = _prob(mid, delta, n, m, i1, i2)
        if probmid >= alpha:
            imax = mid
        else:
            imin = mid

    f[0, 3] = round(imin, pround)
    ls_arr[0, :] = f[0, :].copy()

    partvector = np.round(ls_arr[0, 0] * (m + 2) + ls_arr[0, 1])
    allvector = allvector[~np.isin(allvector, partvector)]

    # From the second table
    morepoint = 1
    kk = 1
    kk1 = 1
    dimoftable = ls_arr.shape[0]

    # Handle special cases
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

    while kk <= dimoftable - 2:
        c = ls_arr[(kk - morepoint) : kk, 0:2].copy()
        s[(kk - morepoint) : kk, :] = c.copy()
        dd = s[:kk, :].copy()
        dd[:, 0] -= 1
        a = dd.copy()
        dd = s[:kk, :].copy()
        dd[:, 1] += 1
        b = dd.copy()

        # Generate N
        n_arr = np.unique(np.vstack((a, b)), axis=0)
        nvector = (n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1] + 1
        nvector = nvector[np.isin(nvector, allvectormove)]

        skvector = (s[:kk, 0] + 1) * (m + 3) + s[:kk, 1] + 1
        nvectortemp = nvector[~np.isin(nvector, skvector)]

        if len(nvectortemp) == 0:
            break

        ntemp = np.zeros((len(nvectortemp), 2))
        ntemp[:, 1] = np.mod(nvectortemp, m + 3)
        ntemp[:, 0] = (nvectortemp - ntemp[:, 1]) / (m + 3) - 1
        ntemp[:, 1] -= 1
        n_arr = ntemp.copy()

        # Generate NC
        nvector = (n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1] + 1
        nvector1 = (n_arr[:, 0] + 2) * (m + 3) + n_arr[:, 1] + 1
        nvector2 = (n_arr[:, 0] + 1) * (m + 3) + n_arr[:, 1]
        drop = np.isin(nvector1, nvector) * 1 + np.isin(nvector2, nvector) * 1
        m_arr = np.column_stack((n_arr, drop))
        mm = m_arr[m_arr[:, 2] < 0.5, :]
        nc_arr = mm.copy()

        if nc_arr.size <= 3:
            length_nc = 1
            nmn = np.full((2, 3), 100)
            if nc_arr.size > 0:
                nmn[0, : len(nc_arr.flatten())] = nc_arr.flatten()
            nc_arr = nmn.copy()
        else:
            length_nc = len(nc_arr)
            if length_nc == 0:
                break

        for i in range(min(length_nc, len(nc_arr))):
            imax = 1 - 100 * delta
            imin = -1 + delta

            if i >= len(nc_arr):
                continue

            ls_arr[kk, 0:2] = nc_arr[i, 0:2].copy()
            i1 = ls_arr[: (kk + 1), 0].copy()
            i2 = ls_arr[: (kk + 1), 1].copy()

            if kk > (dimoftable - 2) / 2:
                partvector = np.round(nc_arr[i, 0] * (m + 2) + nc_arr[i, 1])
                leftvector = allvector[~np.isin(allvector, partvector)]
                if len(leftvector) > 0:
                    i2 = np.mod(leftvector, m + 2)
                    i1 = (leftvector - i2) / (m + 2)

            imax = ls_arr[kk - 1, 3]
            imin = -1 + delta
            if i == 0:
                ncmax = imin

            if kk <= (dimoftable - 2) / 2:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    probmid = _prob2step(mid, delta, n, m, i1, i2, grid_one, grid_two)
                    if probmid >= alpha:
                        imax = mid
                    else:
                        imin = mid
                nc_arr[i, 2] = round(imin, 2)
                if imax >= ncmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        probmid = _prob2step(mid, delta, n, m, i1, i2, grid_one, grid_two)
                        if probmid >= alpha:
                            imax = mid
                        else:
                            imin = mid
                    nc_arr[i, 2] = round(imin, pround)
                ncmax = max(ncmax, imin)
            else:
                while abs(imax - imin) >= 0.1:
                    mid = (imax + imin) / 2
                    probmid = _prob2steplmin(mid, delta, n, m, i1, i2, grid_one, grid_two)
                    if probmid >= 1 - alpha:
                        imin = mid
                    else:
                        imax = mid
                nc_arr[i, 2] = round(imin, 2)
                if imax >= ncmax:
                    while abs(imax - imin) >= precision:
                        mid = (imax + imin) / 2
                        probmid = _prob2steplmin(mid, delta, n, m, i1, i2, grid_one, grid_two)
                        if probmid >= 1 - alpha:
                            imin = mid
                        else:
                            imax = mid
                    nc_arr[i, 2] = round(imin, pround)
                ncmax = max(ncmax, imin)

        if i >= 1 and nc_arr.size > 0:
            valid_rows = ~np.isnan(nc_arr[:, 0])
            if np.any(valid_rows):
                ncnomiss = nc_arr[valid_rows, :].copy()
                ncnomiss = ncnomiss[(-ncnomiss[:, 2]).argsort(), :]  # Sort in descending order
                morepoint = np.sum(ncnomiss[:, 2] >= ncnomiss[0, 2] - delta)

                if morepoint >= 2:
                    ls_arr[kk : (kk + morepoint), 0:2] = ncnomiss[:morepoint, 0:2].copy()
                    ncres = _morepointlsest(kk + morepoint, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two)
                    ls_arr[kk : (kk + morepoint), 3] = ncres

                    for iq in range(morepoint):
                        partvector = np.round(ls_arr[kk + iq, 0] * (m + 2) + ls_arr[kk + iq, 1])
                        allvector = allvector[~np.isin(allvector, partvector)]
                    kk += morepoint
                else:
                    ls_arr[kk, 0:2] = ncnomiss[0, 0:2].copy()
                    ls_arr[kk, 3] = ncnomiss[0, 2].copy()
                    partvector = np.round(ls_arr[kk, 0] * (m + 2) + ls_arr[kk, 1])
                    allvector = allvector[~np.isin(allvector, partvector)]
                    kk += 1
        elif nc_arr.size > 0:
            ls_arr[kk, 0:2] = nc_arr[0, 0:2].copy()
            ls_arr[kk, 3] = nc_arr[0, 2].copy()
            partvector = np.round(ls_arr[kk, 0] * (m + 2) + ls_arr[kk, 1])
            allvector = allvector[~np.isin(allvector, partvector)]
            kk += 1
        else:
            kk += 1

        if sides == ConfidenceIntervalSides.LeftSided.value:
            if np.sum(np.isin(datavector_l, allvector)) == 0:
                for jj in range(kk1, kk):
                    if ls_arr[jj, 0] == x and ls_arr[jj, 1] == y:
                        output[1] = ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    ncres = _morepointlsest(kk, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two)
                    output[1] = ncres
                output[2] = 1
                kk = dimoftable
        else:
            if np.sum(np.isin(datavector_u, allvector)) == 0:
                for jj in range(kk1, kk):
                    if ls_arr[jj, 0] == (n - x) and ls_arr[jj, 1] == (m - y):
                        output[2] = -ls_arr[jj, 3]
                if kk >= kk1 + 2 - delta:
                    ncres = _morepointlsest(kk, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two)
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

    part1 = np.log(comb(n, i1))[:, np.newaxis] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))

    part2 = np.log(comb(m, i2))[:, np.newaxis] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    expsum = np.exp(part1 + part2)
    sumofprob = np.sum(expsum, axis=0)
    return np.max(sumofprob)


def _prob2step(delv, delta, n, m, i1, i2, grid_one, grid_two):
    """Two-step probability calculation with grid refinement"""
    if delv < 0:
        p0 = np.linspace(-delv + delta, 1 - delta, grid_one)
    else:
        p0 = np.linspace(delta, 1 - delv - delta, grid_one)

    i1 = np.atleast_1d(i1)
    i2 = np.atleast_1d(i2)

    part1 = np.log(comb(n, i1))[:, np.newaxis] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))

    part2 = np.log(comb(m, i2))[:, np.newaxis] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    expsum = np.exp(part1 + part2)
    sumofprob = np.sum(expsum, axis=0)
    # mansum = np.max(sumofprob)

    # Refine the grid around the maximum
    stepv = (p0[-1] - p0[0]) / grid_one
    maxloc = np.argmax(sumofprob)
    lowerb = max(p0[0], p0[maxloc] - stepv) + delta
    upperb = min(p0[-1], p0[maxloc] + stepv) - delta

    p0 = np.linspace(lowerb, upperb, grid_two)

    part1 = np.log(comb(n, i1))[:, np.newaxis] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))

    part2 = np.log(comb(m, i2))[:, np.newaxis] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    expsum = np.exp(part1 + part2)
    sumofprob = np.sum(expsum, axis=0)
    return np.max(sumofprob)


def _prob2steplmin(delv, delta, n, m, i1, i2, grid_one, grid_two):
    """Two-step minimum probability calculation with grid refinement"""
    if delv < 0:
        p0 = np.linspace(-delv + delta, 1 - delta, grid_one)
    else:
        p0 = np.linspace(delta, 1 - delv - delta, grid_one)

    i1 = np.atleast_1d(i1)
    i2 = np.atleast_1d(i2)

    part1 = np.log(comb(n, i1))[:, np.newaxis] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))

    part2 = np.log(comb(m, i2))[:, np.newaxis] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    expsum = np.exp(part1 + part2)
    sumofprob = np.sum(expsum, axis=0)
    # mansum = np.min(sumofprob)

    # Refine the grid around the minimum
    stepv = (p0[-1] - p0[0]) / grid_one
    minloc = np.argmin(sumofprob)
    lowerb = max(p0[0], p0[minloc] - stepv) + delta
    upperb = min(p0[-1], p0[minloc] + stepv) - delta

    p0 = np.linspace(lowerb, upperb, grid_two)

    part1 = np.log(comb(n, i1))[:, np.newaxis] + np.outer(i1, np.log(p0 + delv)) + np.outer(n - i1, np.log(1 - p0 - delv))

    part2 = np.log(comb(m, i2))[:, np.newaxis] + np.outer(i2, np.log(p0)) + np.outer(m - i2, np.log(1 - p0))

    expsum = np.exp(part1 + part2)
    sumofprob = np.sum(expsum, axis=0)
    return np.min(sumofprob)


def _morepointlsest(morekk, delta, precision, pround, alpha, n, m, ls_arr, grid_one, grid_two):
    """Calculate confidence interval bounds with multiple points"""
    i1 = ls_arr[:morekk, 0].copy()
    i2 = ls_arr[:morekk, 1].copy()

    imax = 1 - 100 * delta
    imin = -1 + delta

    while abs(imax - imin) >= precision:
        mid = (imax + imin) / 2
        probmid = _prob2step(mid, delta, n, m, i1, i2, grid_one, grid_two)
        if probmid >= alpha:
            imax = mid
        else:
            imin = mid

    return round(imin, pround)
