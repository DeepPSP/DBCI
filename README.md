# Confidence Intervals for Difference of Binomial Proportions

![pytest](https://github.com/DeepPSP/DBCI/actions/workflows/run-pytest.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/dbci?style=flat-square)
![downloads](https://img.shields.io/pypi/dm/dbci?style=flat-square)
![license](https://img.shields.io/github/license/DeepPSP/DBCI?style=flat-square)

Computation of confidence intervals for binomial proportions and for difference of binomial proportions.

## References

1. <a name="ref1"></a> [SAS](https://www.lexjansen.com/wuss/2016/127_Final_Paper_PDF.pdf)
2. <a name="ref2"></a> [PASS](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Difference_Between_Two_Proportions.pdf)
3. <a name="ref3"></a> [statsmodels.stats.proportion](https://www.statsmodels.org/devel/_modules/statsmodels/stats/proportion.html)
4. <a name="ref4"></a> [scipy.stats._binomtest](https://github.com/scipy/scipy/blob/main/scipy/stats/_binomtest.py)
5. <a name="ref5"></a> [corplingstats](https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/)
6. <a name="ref6"></a> [DescTools.StatsAndCIs](https://github.com/AndriSignorell/DescTools/blob/master/R/StatsAndCIs.r)

## NOTE

[Reference 1](#ref1) has errors in the description of the methods `Wilson CC`, `Mee`, `Miettinen-Nurminen`.
The correct computation of `Wilson CC` is given in [Reference 5](#ref5).
The correct computation of `Mee`, `Miettinen-Nurminen` are given in the **code blocks** in [Reference 1](#ref1)

## Test data

[Test data](test/test-data/) are

1. taken from [Reference 1](#ref1) for automatic test of the correctness of the implementation of the algorithms.
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

The filenames has the following pattern:

```python
# for computing confidence interval for difference of binomial proportions
"example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)-vs-(?P<ref_positive>[\\d]+)-(?P<ref_total>[\\d]+)\\.csv"

# for computing confidence interval for binomial proportions
"example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)\\.csv"
```
