# Confidence Intervals for Difference of Binomial Proportions

![pytest](https://github.com/DeepPSP/DBCI/actions/workflows/run-pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/DeepPSP/DBCI/branch/master/graph/badge.svg?token=4IQD228F7L)](https://codecov.io/gh/DeepPSP/DBCI)
![PyPI](https://img.shields.io/pypi/v/diff-binom-confint?style=flat-square)
![downloads](https://img.shields.io/pypi/dm/diff-binom-confint?style=flat-square)
![license](https://img.shields.io/github/license/DeepPSP/DBCI?style=flat-square)

Computation of confidence intervals for binomial proportions and for difference of binomial proportions.

## Installation

Run

```bash
python -m pip install diff-binom-confint
```

or install the latest version in [GitHub](https://github.com/DeepPSP/DBCI/) using

```bash
python -m pip install git+https://github.com/DeepPSP/DBCI.git
```

or git clone this repository and install locally via

```bash
cd DBCI
python -m pip install .
```

## `Numba` accelerated version

Install using

```bash
python -m pip install diff-binom-confint[acc]
```

## Usage examples

```python
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
```

## Implemented methods

### Confidence intervals for binomial proportions

<details>
<summary>Click to view!</summary>

|   Method (type)   | Implemented        |
|-------------------|--------------------|
| wilson            | :heavy_check_mark: |
| wilson-cc         | :heavy_check_mark: |
| wald              | :heavy_check_mark: |
| wald-cc           | :heavy_check_mark: |
| agresti-coull     | :heavy_check_mark: |
| jeffreys          | :heavy_check_mark: |
| clopper-pearson   | :heavy_check_mark: |
| arcsine           | :heavy_check_mark: |
| logit             | :heavy_check_mark: |
| pratt             | :heavy_check_mark: |
| witting           | :heavy_check_mark: |
| mid-p             | :heavy_check_mark: |
| lik               | :heavy_check_mark: |
| blaker            | :heavy_check_mark: |
| modified-wilson   | :heavy_check_mark: |
| modified-jeffreys | :heavy_check_mark: |

</details>

### Confidence intervals for difference of binomial proportions

<details>
<summary>Click to view!</summary>

|   Method (type)             | Implemented        |
|-----------------------------|--------------------|
| wilson                      | :heavy_check_mark: |
| wilson-cc                   | :heavy_check_mark: |
| wald                        | :heavy_check_mark: |
| wald-cc                     | :heavy_check_mark: |
| haldane                     | :heavy_check_mark: |
| jeffreys-perks              | :heavy_check_mark: |
| mee                         | :heavy_check_mark: |
| miettinen-nurminen          | :heavy_check_mark: |
| true-profile                | :heavy_check_mark: |
| hauck-anderson              | :heavy_check_mark: |
| agresti-caffo               | :heavy_check_mark: |
| carlin-louis                | :heavy_check_mark: |
| brown-li                    | :heavy_check_mark: |
| brown-li-jeffrey            | :heavy_check_mark: |
| miettinen-nurminen-brown-li | :heavy_check_mark: |
| exact                       | :x:                |
| mid-p                       | :x:                |
| santner-snell               | :x:                |
| chan-zhang                  | :x:                |
| agresti-min                 | :x:                |
| wang                        | :x:                |
| pradhan-banerjee            | :x:                |

</details>

## References

1. <a name="ref1"></a> [SAS](https://www.lexjansen.com/wuss/2016/127_Final_Paper_PDF.pdf)
2. <a name="ref2"></a> [PASS](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Difference_Between_Two_Proportions.pdf)
3. <a name="ref3"></a> [statsmodels.stats.proportion](https://www.statsmodels.org/devel/_modules/statsmodels/stats/proportion.html)
4. <a name="ref4"></a> [scipy.stats._binomtest](https://github.com/scipy/scipy/blob/main/scipy/stats/_binomtest.py)
5. <a name="ref5"></a> [corplingstats](https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/)
6. <a name="ref6"></a> [DescTools.StatsAndCIs](https://github.com/AndriSignorell/DescTools/blob/master/R/StatsAndCIs.r)
7. <a name="ref7"></a> [Newcombee](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I)

## NOTE

[Reference 1](#ref1) has errors in the description of the methods `Wilson CC`, `Mee`, `Miettinen-Nurminen`.
The correct computation of `Wilson CC` is given in [Reference 5](#ref5).
The correct computation of `Mee`, `Miettinen-Nurminen` are given in the **code blocks** in [Reference 1](#ref1)

## Test data

[Test data](test/test-data/) are

1. taken (with slight modification, e.g. the `upper_bound` of `miettinen-nurminen-brown-li` method in the [edge case file](test/test-data/example-10-10-vs-0-20.csv)) from [Reference 1](#ref1) for automatic test of the correctness of the implementation of the algorithms.
2. generated using [DescTools.StatsAndCIs](#ref6) via

    ```R
    library("DescTools")
    library("data.table")

    results = data.table()
    for (m in c("wilson", "wald", "waldcc", "agresti-coull", "jeffreys",
                    "modified wilson", "wilsoncc","modified jeffreys",
                    "clopper-pearson", "arcsine", "logit", "witting", "pratt", 
                    "midp", "lik", "blaker")){
        ci = BinomCI(84,101,method = m)
        new_row = data.table("method" = m, "ratio"=ci[1], "lower_bound" = ci[2], "upper_bound" = ci[3])
        results = rbindlist(list(results, new_row))
    }
    fwrite(results, "./test/test-data/example-84-101.csv")  # with manual slight adjustment of method names
    ```

3. taken from [Reference 7](#ref7) (Table II).

The filenames has the following pattern:

```python
# for computing confidence interval for difference of binomial proportions
"example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)-vs-(?P<ref_positive>[\\d]+)-(?P<ref_total>[\\d]+)\\.csv"

# for computing confidence interval for binomial proportions
"example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)\\.csv"
```

Note that the out-of-range values (e.g. `> 1`) are left as empty values in the `.csv` files.

## Known Issues

1. Edge cases incorrect for the method `true-profile`.
