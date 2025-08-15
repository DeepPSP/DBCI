.. DBCI documentation master file, created by
   sphinx-quickstart on Wed Mar  1 23:45:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DBCI's documentation!
================================

A package for computing confidence intervals for binomial proportions,
and confidence intervals for the difference of binomial proportions.

Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

The package can be installed via `PyPI` using

.. code:: bash

   python -m pip install diff-binom-confint

or one can install the latest version on \ `GitHub <https://github.com/DeepPSP/DBCI/>`__\  using

.. code:: bash

   python -m pip install git+https://github.com/DeepPSP/DBCI.git


or clone this repository and install locally via

.. code:: bash

   git clone https://github.com/DeepPSP/DBCI.git
   cd DBCI
   python -m pip install .

Numba accelerated version
~~~~~~~~~~~~~~~~~~~~~~~~~~

One can install the \ `Numba <https://numba.pydata.org/>`__\  accelerated version of the package using

.. code:: bash

   python -m pip install diff-binom-confint[acc]

Streamlit app
^^^^^^^^^^^^^^^

One can also use the \ `Streamlit app <https://diff-binom-confint.streamlit.app>`_\  to compute confidence intervals for binomial proportions,
and even upload a categorized table of data to obtain a report of risk report for binomial confidence intervals.

Usage examples
^^^^^^^^^^^^^^^

The following example shows how to compute the confidence interval for the
difference of two binomial proportions using the Wilson method.

.. code-block:: python

    from diff_binom_confint import compute_difference_confidence_interval

    n_positive, n_total = 84, 101
    ref_positive, ref_total = 89, 105

    confint = compute_difference_confidence_interval(
        n_positive,
        n_total,
        ref_positive,
        ref_total,
        conf_level=0.95,
        method="wilson",
    )

Implemented methods
^^^^^^^^^^^^^^^^^^^^

Confidence intervals for binomial proportions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We list the implemented methods for confidence intervals
for binomial proportions in the following table.

+-------------------+----------------------+
| Method (type)     | Implemented          |
+===================+======================+
| wilson            | |:heavy_check_mark:| |
+-------------------+----------------------+
| wilson-cc         | |:heavy_check_mark:| |
+-------------------+----------------------+
| wald              | |:heavy_check_mark:| |
+-------------------+----------------------+
| wald-cc           | |:heavy_check_mark:| |
+-------------------+----------------------+
| agresti-coull     | |:heavy_check_mark:| |
+-------------------+----------------------+
| jeffreys          | |:heavy_check_mark:| |
+-------------------+----------------------+
| clopper-pearson   | |:heavy_check_mark:| |
+-------------------+----------------------+
| arcsine           | |:heavy_check_mark:| |
+-------------------+----------------------+
| logit             | |:heavy_check_mark:| |
+-------------------+----------------------+
| pratt             | |:heavy_check_mark:| |
+-------------------+----------------------+
| witting           | |:heavy_check_mark:| |
+-------------------+----------------------+
| mid-p             | |:heavy_check_mark:| |
+-------------------+----------------------+
| lik               | |:heavy_check_mark:| |
+-------------------+----------------------+
| blaker            | |:heavy_check_mark:| |
+-------------------+----------------------+
| modified-wilson   | |:heavy_check_mark:| |
+-------------------+----------------------+
| modified-jeffreys | |:heavy_check_mark:| |
+-------------------+----------------------+

Confidence intervals for difference of binomial proportions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is the table of implemented methods for computing
confidence intervals for the difference of binomial proportions.

+-----------------------------+----------------------+
| Method (type)               | Implemented          |
+=============================+======================+
| wilson                      | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| wilson-cc                   | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| wald                        | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| wald-cc                     | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| haldane                     | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| jeffreys-perks              | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| mee                         | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| miettinen-nurminen          | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| true-profile                | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| hauck-anderson              | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| agresti-caffo               | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| carlin-louis                | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| brown-li                    | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| brown-li-jeffrey            | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| miettinen-nurminen-brown-li | |:heavy_check_mark:| |
+-----------------------------+----------------------+
| exact                       | |:x:|                |
+-----------------------------+----------------------+
| mid-p                       | |:x:|                |
+-----------------------------+----------------------+
| santner-snell               | |:x:|                |
+-----------------------------+----------------------+
| chan-zhang                  | |:x:|                |
+-----------------------------+----------------------+
| agresti-min                 | |:x:|                |
+-----------------------------+----------------------+
| wang                        | |:x:|                |
+-----------------------------+----------------------+
| pradhan-banerjee            | |:x:|                |
+-----------------------------+----------------------+

.. toctree::
   :caption: API reference
   :maxdepth: 1

   api

.. toctree::
   :caption: Advanced topics
   :maxdepth: 1

   advanced

References
^^^^^^^^^^

1. `SAS <https://github.com/DeepPSP/DBCI/blob/master/references/Constructing%20Confidence%20Intervals%20for%20the%20Differences%20of%20Binomial%20Proportions%20in%20SAS.pdf>`_
2. `PASS <https://github.com/DeepPSP/DBCI/blob/master/references/Confidence%20Intervals%20for%20the%20Difference%20Between%20Two%20Proportions%20-%20PASS.pdf>`_
3. `statsmodels.stats.proportion <https://www.statsmodels.org/devel/_modules/statsmodels/stats/proportion.html>`_
4. `scipy.stats._binomtest <https://github.com/scipy/scipy/blob/main/scipy/stats/_binomtest.py>`_
5. `corplingstats <https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/>`_
6. `DescTools.StatsAndCIs <https://github.com/AndriSignorell/DescTools/blob/master/R/StatsAndCIs.r>`_

..
   7. `Newcombee <https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
