Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.1.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

`Unreleased <https://github.com/DeepPSP/DBCI/compare/v0.1.0...HEAD>`__
----------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.1.0 <https://github.com/DeepPSP/DBCI/compare/v0.0.17...v0.1.0>`__ - 2025-8-15
-----------------------------------------------------------------------------------

Added
~~~~~

- Add the `wang` method for computing confidence intervals for the
  difference of two proportions in `#8 <https://github.com/DeepPSP/DBCI/pull/8>`__.

`0.0.17 <https://github.com/DeepPSP/DBCI/compare/v0.0.16...v0.0.17>`__ - 2024-7-30
-----------------------------------------------------------------------------------

Fixed
~~~~~

- Fix errors in the witting method
  `# 7 <https://github.com/DeepPSP/DBCI/pull/7>`__.

`0.0.16 <https://github.com/DeepPSP/DBCI/compare/v0.0.15...v0.0.16>`__ - 2024-5-19
-----------------------------------------------------------------------------------

Changed
~~~~~~~

- Enhance the streamlit APP.
- Increase the default value for digits from 5 to 7.

Fixed
~~~~~

- Fix bugs for the cases when ratio = 0 or 1 for logit type binomial confint.
- Fix error for the `hauck-anderson` method.
- Fix errors for edge cases for the witting method.

`0.0.15 <https://github.com/DeepPSP/DBCI/compare/v0.0.14...v0.0.15>`__ - 2023-10-26
-----------------------------------------------------------------------------------

Added
~~~~~

- Add python 3.11 to supported python versions.
- Add streamlit APP for the package.
- Add method ``asdict``, ``astable`` for the ``ConfidenceInterval`` class.

`0.0.14 <https://github.com/DeepPSP/DBCI/compare/v0.0.13...v0.0.14>`__ - 2023-10-4
-----------------------------------------------------------------------------------

Added
~~~~~

- Add massive random test
  `# 4 <https://github.com/DeepPSP/DBCI/pull/4>`__.
- Add function ``make_risk_report``.

`0.0.13 <https://github.com/DeepPSP/DBCI/compare/v0.0.12...v0.0.13>`__ - 2023-4-26
-----------------------------------------------------------------------------------

Fixed
~~~~~

- Fix ``ZeroDivisionError`` encountered using `numba` acceleration
  `# 2 <https://github.com/DeepPSP/DBCI/pull/2>`__.

`0.0.12 <https://github.com/DeepPSP/DBCI/compare/v0.0.11...v0.0.12>`__ - 2022-12-10
-----------------------------------------------------------------------------------

Changed
~~~~~~~

- Update error/warning messages.

Fixed
~~~~~

- Fix bugs in function ``compute_confidence_interval``.

`0.0.11 <https://github.com/DeepPSP/DBCI/compare/v0.0.10...v0.0.11>`__ - 2022-11-21
-----------------------------------------------------------------------------------

Added
~~~~~

- Add `Carlin` and `Louis` methods.
- Add code coverage check in the pytest action.

`0.0.10 <https://github.com/DeepPSP/DBCI/compare/v0.0.9...v0.0.10>`__ - 2022-10-6
-----------------------------------------------------------------------------------

Fixed
~~~~~

- Fix errors for ``compute_difference_confidence_interval`` using method
  `true-profile` when any of the 4 cells is zero.

`0.0.9 <https://github.com/DeepPSP/DBCI/compare/v0.0.8...v0.0.9>`__ - 2022-9-19
-----------------------------------------------------------------------------------

Added
~~~~~

- Add `numba` accelerator.

`0.0.8 <https://github.com/DeepPSP/DBCI/compare/v0.0.7...v0.0.8>`__ - 2022-9-18
-----------------------------------------------------------------------------------

Changed
~~~~~~~

- Update the computation of confidence interval using method `wilson-cc`.
- update the computation of difference proportions confidence interval
  using methods `wilson-cc`, `mee`, and `miettinen-nurminen`.

`0.0.7 <https://github.com/DeepPSP/DBCI/compare/v0.0.6...v0.0.7>`__ - 2022-9-16
-----------------------------------------------------------------------------------

Fixed
~~~~~

- Fix bugs in the computation of 1-sided confidence intervals.

`0.0.6 <https://github.com/DeepPSP/DBCI/compare/v0.0.5...v0.0.6>`__ - 2022-9-14
-----------------------------------------------------------------------------------

Fixed
~~~~~

- Fix bugs in ``compute_confidence_interval`` for 1-sided cases.

`0.0.5 <https://github.com/DeepPSP/DBCI/compare/v0.0.4...v0.0.5>`__ - 2022-9-14
-----------------------------------------------------------------------------------

Added
~~~~~

- Add attribute ``estimate`` to the class ``ConfidenceInterval``.
- Add parameter ``sides`` for the functions for computing confidence intervals.

.. note::

    This release was YANKED.

Changed
~~~~~~~

- Replace field ``type`` with ``method`` for class ``ConfidenceInterval``.

.. note::

    This release was YANKED.

`0.0.4 <https://github.com/DeepPSP/DBCI/compare/v0.0.3...v0.0.4>`__ - 2022-9-11
-----------------------------------------------------------------------------------

Changed
~~~~~~~

- Replace keyword argument ``confint_type`` with ``method``,
  keeping in accordance with conventional terminologies.

`0.0.3 <https://github.com/DeepPSP/DBCI/compare/v0.0.2...v0.0.3>`__ - 2022-9-9
-----------------------------------------------------------------------------------

Added
~~~~~

- Add parameter ``clip`` for the functions for computing confidence intervals.
- Add `blaker` method for ``compute_confidence_interval``.
- Add `lik` method for ``compute_confidence_interval``.
- Add `mid-p` method for ``compute_confidence_interval``.
- Add `modified-wilson` method for ``compute_confidence_interval``.
- Add `witting` method for ``compute_confidence_interval``.

`0.0.2 <https://github.com/DeepPSP/DBCI/releases/tag/v0.0.2>`__ - 2022-9-8
-----------------------------------------------------------------------------------

Update README.md

`0.0.1 <https://pypi.org/project/diff-binom-confint/0.0.1/>`__ - 2022-9-7
-----------------------------------------------------------------------------------

Initial release.

Implements most methods in
`DescTools.StatsAndCIs <https://github.com/AndriSignorell/DescTools/blob/master/R/StatsAndCIs.r>`__
for computing confidence intervals for a proportion or the difference of two proportions.
