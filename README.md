# Confidence Intervals for Difference of Binomial Proportions

Computation of confidence intervals for binomial proportions and for difference of binomial proportions.

## References

1. <a name="ref1"></a> [SAS](https://www.lexjansen.com/wuss/2016/127_Final_Paper_PDF.pdf)
2. <a name="ref2"></a> [PASS](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Difference_Between_Two_Proportions.pdf)
3. <a name="ref3"></a> [statsmodels.stats.proportion](https://www.statsmodels.org/devel/_modules/statsmodels/stats/proportion.html)
4. <a name="ref4"></a> [scipy.stats._binomtest](https://github.com/scipy/scipy/blob/main/scipy/stats/_binomtest.py)
5. <a name="ref5"></a> [corplingstats](https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/)

## NOTE

[Reference 1](#ref1) has errors in the description of the methods `Wilson CC`, `Mee`, `Miettinen-Nurminen`.
The correct computation of `Wilson CC` is given in [Reference 5](#ref5).
The correct computation of `Mee`, `Miettinen-Nurminen` are given in the **code blocks** in [Reference 1](#ref1)
