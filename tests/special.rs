use peroxide::fuga::{LambertWAccuracyMode::*, *};
use std::f64::consts::{LN_2, PI};

#[test]
fn lambert_w_test() {
    assert_eq!(lambert_w0(1.0, Precise), 0.567143290409784);
    assert!(nearly_eq(lambert_w0(1.0, Simple), 0.567143290409784));
}

#[test]
fn test_gamma_poles_and_undefined() {
    // Gamma(0) approaches infinity
    assert!(gamma(0.0).is_infinite());
    assert!(gamma(0.0).is_sign_positive());

    // Gamma(-0.0) diverges to negative infinity: tgamma(+0.0) is +inf and
    // tgamma(-0.0) is -inf in C99, and `z == 0.0` matches both zeros.
    assert!(gamma(-0.0).is_infinite());
    assert!(gamma(-0.0).is_sign_negative());

    // Gamma for negative integers is mathematically undefined (diverges)
    assert!(gamma(-1.0).is_nan());
    assert!(gamma(-2.0).is_nan());
    assert!(gamma(-10.0).is_nan());

    // Log-Gamma goes to positive infinity for all poles
    assert!(ln_gamma(0.0).is_infinite());
    assert!(ln_gamma(-1.0).is_infinite());
    assert!(ln_gamma(-10.0).is_infinite());
    assert!(ln_gamma(-0.0).is_infinite());
    assert!(ln_gamma(-0.0).is_sign_positive());
}

#[test]
fn test_gamma_integer_fast_path() {
    // Standard small factorials: Gamma(n) = (n-1)!
    assert_eq!(gamma(1.0), 1.0); // 0!
    assert_eq!(gamma(2.0), 1.0); // 1!
    assert_eq!(gamma(4.0), 6.0); // 3!
    assert_eq!(gamma(5.0), 24.0); // 4!
    assert_eq!(gamma(10.0), 362_880.0); // 9!

    // Wolfram Alpha high-precision check (21!)
    // f64 can exactly represent this without precision loss
    assert_eq!(gamma(22.0), 51_090_942_171_709_440_000.0);

    // Maximum limit of f64 float representation (~171.6)
    // Ensure it doesn't panic on overflow, but correctly yields Infinity
    assert!(gamma(172.0).is_infinite());
}

#[test]
fn test_gamma_positive_floats() {
    let sqrt_pi = PI.sqrt();

    // Gamma(0.5) = sqrt(PI)
    assert!(nearly_eq(gamma(0.5), sqrt_pi));

    // Gamma(1.5) = 0.5 * sqrt(PI)
    assert!(nearly_eq(gamma(1.5), 0.5 * sqrt_pi));

    // Gamma(2.5) = 1.329340388179...
    assert!(nearly_eq(gamma(2.5), 0.75 * sqrt_pi));
}

#[test]
fn test_gamma_negative_floats_reflection() {
    let sqrt_pi = PI.sqrt();

    // Gamma(-0.5) = -2 * sqrt(PI)
    // This validates that .abs() is NOT used on the sine in gamma_approx
    assert!(nearly_eq(gamma(-0.5), -2.0 * sqrt_pi));
    assert!(gamma(-0.5).is_sign_negative());

    // Gamma(-1.5) = (4/3) * sqrt(PI)
    assert!(nearly_eq(gamma(-1.5), (4.0 / 3.0) * sqrt_pi));
    assert!(gamma(-1.5).is_sign_positive());

    // Gamma(-2.5) = -(8/15) * sqrt(PI)
    assert!(nearly_eq(gamma(-2.5), -(8.0 / 15.0) * sqrt_pi));
    assert!(gamma(-2.5).is_sign_negative());
}

#[test]
fn test_ln_gamma_consistency() {
    // ln_gamma(x) should equal ln(|Gamma(x)|) across the board
    let test_values = vec![0.5, 1.5, 2.5, 10.5];

    for &val in &test_values {
        let expected = gamma(val).ln();
        let actual = ln_gamma(val);
        assert!(
            nearly_eq(expected, actual),
            "Failed at positive float: val={}, expected={}, actual={}",
            val,
            expected,
            actual
        );
    }

    // Test Negative Floats to ensure `.abs()` prevents NaN
    let negative_test_values = vec![-0.5, -1.5, -2.5, -10.5];

    for &val in &negative_test_values {
        let expected = gamma(val).abs().ln();
        let actual = ln_gamma(val);
        assert!(
            nearly_eq(expected, actual),
            "Failed at negative float: val={}, expected={}, actual={}",
            val,
            expected,
            actual
        );
    }
}

#[test]
fn test_ln_gamma_exact_at_small_integers() {
    // Gamma(1) = Gamma(2) = 1, so the log is exactly zero. The Lanczos series on
    // its own lands near -5e-12 here, and that leaks into any caller that forms a
    // difference of log-gammas, such as a log-space binomial coefficient.
    assert_eq!(ln_gamma(1.0), 0.0);
    assert_eq!(ln_gamma(2.0), 0.0);

    // Reference values computed with mpmath at 40 digits, rounded to f64. The
    // tolerance is tight enough to fail if the integer path is removed, since the
    // Lanczos series alone is only good to about 4e-13 relative here.
    let cases: [(f64, f64); 3] = [
        (3.0, LN_2),
        (16.0, 27.89927138384089),
        (23.0, 48.47118135183523),
    ];

    for &(z, expected) in &cases {
        let got = ln_gamma(z);
        let rel = (got - expected).abs() / expected.abs();
        assert!(
            rel < 1e-14,
            "ln_gamma({}) = {}, expected {}, relative error {:e}",
            z,
            got,
            expected,
            rel
        );
    }
}

#[test]
fn test_ln_gamma_against_reference() {
    // Independent reference values (mpmath, 40 digits, rounded to f64).
    // test_ln_gamma_consistency compares ln_gamma against gamma().ln(), which is
    // circular for non-integer z >= 0.5 because gamma is ln_gamma(z).exp() there,
    // so these pin the actual values instead. Note nearly_eq compares magnitudes
    // and cannot catch a sign flip, hence the explicit signed comparison.
    let cases: [(f64, f64); 8] = [
        (0.5, 0.5723649429247001),
        (1.5, -0.12078223763524522),
        (2.5, 0.2846828704729192),
        (10.5, 13.940625219403763),
        (-0.5, 1.2655121234846454),
        (-1.5, 0.860047015376481),
        (-2.5, -0.056243716497674054),
        (-10.5, -15.147270590717842),
    ];

    for &(z, expected) in &cases {
        let got = ln_gamma(z);
        let tol = 1e-9 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tol,
            "ln_gamma({}) = {}, expected {} (tolerance {:e})",
            z,
            got,
            expected,
            tol
        );
    }
}
