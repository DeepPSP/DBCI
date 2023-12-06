"""
"""

import random
import time

import pytest

try:
    from numba import njit
except ModuleNotFoundError:
    njit = None

from diff_binom_confint._utils import (
    Accelerator,
    accelerator,
    add_docstring,
    dummy_accelerator,
    remove_parameters_returns_from_docstring,
)


def test_add_docstring():
    @add_docstring("This is a new docstring.")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "This is a new docstring."
    func(1, 2)

    @add_docstring("Leading docstring.", mode="prepend")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "Leading docstring.\nThis is a docstring."
    func(1, 2)

    @add_docstring("Trailing docstring.", mode="append")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "This is a docstring.\nTrailing docstring."
    func(1, 2)

    with pytest.raises(ValueError, match=f"mode {repr('not-supported')} is not supported"):

        @add_docstring("Trailing docstring.", mode="not-supported")
        def func(a, b):
            """This is a docstring."""
            return a + b


def test_remove_parameters_returns_from_docstring():
    new_docstring = remove_parameters_returns_from_docstring(
        remove_parameters_returns_from_docstring.__doc__,
        parameters=["returns_indicator", "parameters_indicator"],
        returns="new_doc",
    )
    assert (
        new_docstring
        == """Remove parameters and/or returns from docstring,
    which is of the format of numpydoc.

    Parameters
    ----------
    doc : str
        The docstring to be processed.
    parameters : str or list of str, default None
        Parameters to be removed.
    returns : str or list of str, default None
        Returned values to be removed.

    Returns
    -------
    None

    """
    )

    new_docstring = remove_parameters_returns_from_docstring(
        remove_parameters_returns_from_docstring.__doc__,
    )
    assert new_docstring == remove_parameters_returns_from_docstring.__doc__


@accelerator.accelerator
def monte_carlo_pi_acc(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


def test_accelerator():
    acc = Accelerator()

    acc.set_accelerator(None)
    assert acc.accelerator == dummy_accelerator

    acc.set_accelerator("numba")
    assert acc.accelerator == (njit or dummy_accelerator)

    # call `monte_carlo_pi_acc` one time
    # to allow `numba` to compile the function
    monte_carlo_pi_acc(1000)

    nsamples = int(1e8)

    start = time.time()
    monte_carlo_pi_acc(nsamples)
    acc_time = time.time() - start

    start = time.time()
    monte_carlo_pi(nsamples)
    no_acc_time = time.time() - start

    if njit is not None:
        assert acc_time < no_acc_time / 5

    with pytest.raises(ValueError, match=f"Accelerator {repr('taichi')} is not supported"):
        acc.set_accelerator("taichi")
