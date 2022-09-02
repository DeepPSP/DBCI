"""
"""

from pathlib import Path

from pytest import approx

try:
    from dbci import compute_difference_confidence_interval
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src import compute_difference_confidence_interval


def test_difference_confidence_interval():
    """
    A CASE STUDY: EXAMPLE FROM AN HIV CLINICAL TRIAL

    Consider Week 48 results of the PROGRESS trial as presented by Reynes (2010).
    There, the proportion of patients who were responders on a new treatment was 84/101 (83.2%)
    while the proportion of patients who were responders on the standard of care was 89/105 (84.8%).
    Non-inferiority (NI) testing was to be performed by constructing a confidence interval for the difference in proportions.
    If the lower bound of the confidence interval is above a pre-defined margin (eg, -12%), then NI can be concluded.
    The table below shows what the confidence intervals from the various methods above (excluding Methods 8 and 9) would be.

    Reference
    ---------
    Reynes JLA, Pulido F, et al., “Lopinavir/ritonavir combined with raltegravir demonstrated similar antiviral efficacy and safety as lopinavir/ritonavir combined with tenofovir disoproxil fumarate/emtricitabine in treatment-naïve HIV-1 infected subjects.” Program and abstracts of the XVIII International AIDS Conference; July 18-23, 2010; Vienna, Austria. Abstract MOAB0101.
    """
    n_positive, n_negative = 84, 101 - 84
    ref_positive, ref_negative = 89, 105 - 89

    # Wald, no CC
    lower_bound, upper_bound = compute_difference_confidence_interval(
        n_positive, n_negative, ref_positive, ref_negative, confint_type="wald"
    ).astuple()

    assert lower_bound == approx(-0.1162, abs=1e-4)
    assert upper_bound == approx(0.0843, abs=1e-4)

    # Wald, CC
    lower_bound, upper_bound = compute_difference_confidence_interval(
        n_positive, n_negative, ref_positive, ref_negative, confint_type="wald_cc"
    ).astuple()

    assert lower_bound == approx(-0.1259, abs=1e-4)
    assert upper_bound == approx(0.0940, abs=1e-4)

    # Haldane
    lower_bound, upper_bound = compute_difference_confidence_interval(
        n_positive, n_negative, ref_positive, ref_negative, confint_type="haldane"
    ).astuple()

    assert lower_bound == approx(-0.1152, abs=1e-4)
    assert upper_bound == approx(0.0834, abs=1e-4)

    # Jeffreys-Perks
    lower_bound, upper_bound = compute_difference_confidence_interval(
        n_positive,
        n_negative,
        ref_positive,
        ref_negative,
        confint_type="jeffreys-perks",
    ).astuple()

    assert lower_bound == approx(-0.1160, abs=1e-4)
    assert upper_bound == approx(0.0843, abs=1e-4)

    # Mee, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="mee"
    # ).astuple()

    # assert lower_bound == approx(-0.1188, abs=1e-4)
    # assert upper_bound == approx(0.0857, abs=1e-4)

    # Miettinen-Nurminen, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive,
    #     n_negative,
    #     ref_positive,
    #     ref_negative,
    #     confint_type="miettinen-nurminen",
    # ).astuple()

    # assert lower_bound == approx(-0.1191, abs=1e-4)
    # assert upper_bound == approx(0.0860, abs=1e-4)

    # Wilson, no CC
    lower_bound, upper_bound = compute_difference_confidence_interval(
        n_positive, n_negative, ref_positive, ref_negative, confint_type="wilson"
    ).astuple()

    assert lower_bound == approx(-0.1177, abs=1e-4)
    assert upper_bound == approx(0.0851, abs=1e-4)

    # Wilson, CC, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="wilson_cc"
    # ).astuple()

    # assert lower_bound == approx(-0.1245, abs=1e-4)
    # assert upper_bound == approx(0.0918, abs=1e-4)

    # True profile, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="true-profile"
    # ).astuple()

    # assert lower_bound == approx(-0.1176, abs=1e-4)
    # assert upper_bound == approx(0.0849, abs=1e-4)

    # Hauck-Andersen, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="hauck-anderson"
    # ).astuple()

    # assert lower_bound == approx(-0.1216, abs=1e-4)
    # assert upper_bound == approx(0.0898, abs=1e-4)

    # Agresti-Caffo, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="agresti-caffo"
    # ).astuple()

    # assert lower_bound == approx(-0.1168, abs=1e-4)
    # assert upper_bound == approx(0.0850, abs=1e-4)

    # Santner-Snell, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="santner-snell"
    # ).astuple()

    # assert lower_bound == approx(-0.1514, abs=1e-4)
    # assert upper_bound == approx(0.1219, abs=1e-4)

    # Chan-Zhang, NOT implemented yet
    # lower_bound, upper_bound = compute_difference_confidence_interval(
    #     n_positive, n_negative, ref_positive, ref_negative, confint_type="chan-zhang"
    # ).astuple()

    # assert lower_bound == approx(-0.1227, abs=1e-4)
    # assert upper_bound == approx(0.0869, abs=1e-4)

    # Brown-Li, NOT implemented yet

    # Miettinen-Nurminen-Brown-Li, NOT implemented yet


if __name__ == "__main__":
    test_difference_confidence_interval()
