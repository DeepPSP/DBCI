"""
"""

import re
import warnings
from typing import Callable, List, Optional, Union

try:
    from numba import njit
except ModuleNotFoundError:
    njit = None
# try:
#     from taichi import kernel as ti_kernel, init as ti_init
#     ti_init()
# except ModuleNotFoundError:
#     ti_kernel = None


__all__ = [
    "add_docstring",
    "remove_parameters_returns_from_docstring",
    "accelerator",
]


def add_docstring(doc: str, mode: str = "replace") -> Callable:
    """Decorator to add docstring to a function or a class.

    Parameters
    ----------
    doc : str
        The docstring to be added.
    mode : str, default "replace"
        The mode of the docstring,
        can be "replace", "append" or "prepend", case insensitive.

    Returns
    -------
    callable
        The function or class with docstring modified.

    """

    def decorator(func_or_cls: Callable) -> Callable:
        pattern = "(\\s^\n){1,}"
        if mode.lower() == "replace":
            func_or_cls.__doc__ = doc
        elif mode.lower() == "append":
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ += new_lines + doc
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ = doc + new_lines + func_or_cls.__doc__
        else:
            raise ValueError(f"mode {repr(mode)} is not supported")
        return func_or_cls

    return decorator


def remove_parameters_returns_from_docstring(
    doc: str,
    parameters: Optional[Union[str, List[str]]] = None,
    returns: Optional[Union[str, List[str]]] = None,
    parameters_indicator: str = "Parameters",
    returns_indicator: str = "Returns",
) -> str:
    """Remove parameters and/or returns from docstring,
    which is of the format of numpydoc.

    Parameters
    ----------
    doc : str
        The docstring to be processed.
    parameters : str or list of str, default None
        Parameters to be removed.
    returns : str or list of str, default None
        Returned values to be removed.
    parameters_indicator : str, default "Parameters"
        The indicator of the parameters section.
    returns_indicator : str, default "Returns"
        The indicator of the returns section.

    Returns
    -------
    new_doc : str
        The processed docstring.

    """
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
    if returns is None:
        returns = []
    elif isinstance(returns, str):
        returns = [returns]

    new_doc = doc.splitlines()
    parameters_indent = None
    returns_indent = None
    start_idx = None
    indices2remove = []
    for idx, line in enumerate(new_doc):
        if line.strip().startswith(parameters_indicator):
            candidate = " " * line.index(parameters_indicator)
            if idx < len(new_doc) - 1 and new_doc[idx + 1] == candidate + "-" * len(parameters_indicator):
                parameters_indent = candidate
        if line.strip().startswith(returns_indicator):
            returns_indent = " " * line.index(returns_indicator)
        if parameters_indent is not None and len(line.lstrip()) == len(line) - len(parameters_indent):
            if any([line.lstrip().startswith(p) for p in parameters]):
                if start_idx is not None:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = idx
            elif start_idx is not None:
                if line.lstrip().startswith(returns_indicator) and len(new_doc[idx - 1].strip()) == 0:
                    indices2remove.extend(list(range(start_idx, idx - 1)))
                else:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = None
        if returns_indent is not None and len(line.lstrip()) == len(line) - len(returns_indent):
            if any([line.lstrip().startswith(p) for p in returns]):
                if start_idx is not None:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = idx
            elif start_idx is not None:
                indices2remove.extend(list(range(start_idx, idx)))
                start_idx = None
    if start_idx is not None:
        indices2remove(list(range(start_idx, len(new_doc))))
        new_doc.extend(["\n", parameters_indicator or returns_indicator])
    new_doc = [line for idx, line in enumerate(new_doc) if idx not in indices2remove]
    # remove trailing empty lines
    idx = max(idx for idx, line in enumerate(new_doc) if len(line.strip()) > 0)
    new_doc = new_doc[: idx + 1]
    if returns_indent is not None:
        if new_doc[-1] == returns_indent + "-" * len(returns_indicator) and new_doc[-2] == returns_indent + returns_indicator:
            new_doc.extend([returns_indent + "None"])
    new_doc.extend(["", returns_indent])
    new_doc = "\n".join(new_doc)
    return new_doc


def dummy_accelerator(func: callable) -> callable:
    """Dummy accelerator."""
    return func


if njit is None:
    njit = dummy_accelerator

# if ti_kernel is None:
#     ti_kernel = dummy_accelerator


class Accelerator(object):
    """Accelerator.

    If `numba` is installed, `numba` is used as the default accelerator.
    Otherwise, a dummy accelerator (no acceleration) is used.
    """

    def __init__(self) -> None:
        if njit is not dummy_accelerator:
            self.accelerator = njit
        # elif ti_kernel is not dummy_accelerator:
        #     self.accelerator = ti_kernel
        else:
            self.accelerator = dummy_accelerator

    def set_accelerator(self, name: Optional[str]) -> None:
        if name is None:
            self.accelerator = dummy_accelerator
        elif name.lower() == "numba":
            if njit is dummy_accelerator:
                warnings.warn("numba is not installed, dummy accelerator is used instead")
            self.accelerator = njit
        # elif name.lower() == "taichi":
        #     if ti_kernel is dummy_accelerator:
        #         warnings.warn(
        #             "`taichi` is not installed, dummy accelerator is used instead"
        #         )
        #     self.accelerator = ti_kernel
        else:
            raise ValueError(f"Accelerator {repr(name)} is not supported")


accelerator = Accelerator()
