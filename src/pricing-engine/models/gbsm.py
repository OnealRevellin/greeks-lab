import time

import numpy as np
from scipy.stats import norm


def value(
    is_call: np.ndarray[bool],
    S: np.ndarray[np.float64],
    K: np.ndarray[np.float64],
    T: np.ndarray[np.float64],
    r: np.ndarray[np.float64],
    sigma: np.ndarray[np.float64],
    b: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """
    Vectorized valuation of European call and put options using the 
    Generalized Black-Scholes-Merton (GBSM) model.

    This implementation follows the formula from:
    Espen Gaarder Haug, "The Complete Guide to Option Pricing Formulas", 2nd ed., page 8.

    The GBSM pricing formula is:

        Call = S * exp((b - r) * T) * N(d1) - K * exp(-r * T) * N(d2)
        Put  = K * exp(-r * T) * N(-d2) - S * exp((b - r) * T) * N(-d1)

    with:
        d1 = [ln(S / K) + (b + 0.5 * sigma²) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Parameters
    ----------
    is_call : np.ndarray[bool]
        Boolean array indicating call (`True`) or put (`False`) option.
    S : np.ndarray[np.float64]
        Spot prices of the underlying asset.
    K : np.ndarray[np.float64]
        Strike prices.
    T : np.ndarray[np.float64]
        Time to maturity (in years).
    r : np.ndarray[np.float64]
        Risk-free interest rates (annualized, continuously compounded).
    sigma : np.ndarray[np.float64]
        Volatilities of the underlying asset.
    b : np.ndarray[np.float64]
        Cost of carry:
            - b = r       → Black-Scholes (non-dividend-paying stock)
            - b = 0       → Futures pricing
            - b = r - q   → Stock with continuous dividend yield q

    Returns
    -------
    np.ndarray
        Option prices for the specified parameters and option type.
    """
    sigma_sqrtT = sigma * np.sqrt(T)
    erT = np.exp(-r * T)
    ebrT = np.exp((b - r) * T)

    d1 = (np.log(S / K) + (b + sigma**2 / 2) * T) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT

    out = np.where(
        is_call,
        S * ebrT * norm.cdf(d1) - K * erT * norm.cdf(d2),
        K * erT * norm.cdf(-d2) - S * ebrT * norm.cdf(-d1)
    )

    return out


if __name__ == "__main__":
    nb_sims = 1_000_000
    flavor = np.random.choice([True, False], nb_sims)
    S = np.random.uniform(80.0, 100.0, nb_sims)
    K = np.random.uniform(80.0, 100.0, nb_sims)
    T = np.random.uniform(0.1, 4.0, nb_sims)
    r = np.random.uniform(0.04, 0.06, nb_sims)
    sigma = np.random.uniform(0.12, 0.65, nb_sims)
    b = np.full(S.shape, 0.0)

    start = time.perf_counter()

    value = value(flavor, S, K, T, r, sigma, b)

    end = time.perf_counter()
    elapsed_us = (end - start) * 1e6

    print(value)
    print(f"Elapsed time: {elapsed_us:.2f} µs")
