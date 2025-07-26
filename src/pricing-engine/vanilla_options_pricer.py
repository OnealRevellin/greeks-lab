from dataclasses import dataclass
import json
import time
from typing import Optional

import numpy as np

from models import gbsm


@dataclass
class VanillaOuputs:
    value: np.ndarray[np.float64]
    delta: np.ndarray[np.float64]
    gamma: np.ndarray[np.float64]
    theta: np.ndarray[np.float64]
    vega: np.ndarray[np.float64]
    rho: np.ndarray[np.float64]


class VanillaOptionsPricer:
    def __init__(
        self,
        model: np.ndarray[str],
        is_call: np.ndarray[bool],
        S: np.ndarray[np.float64],
        K: np.ndarray[np.float64],
        T: np.ndarray[np.float64],
        r: np.ndarray[np.float64],
        sigma: np.ndarray[np.float64],
        b: Optional[np.ndarray[np.float64]] = None,
    ):
        self.model = model
        self.is_call = is_call
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.b = b if b is not None else np.full(S.shape, 0.0)

        self.init_params()
        self.check_inputs()

    def init_params(self) -> None:
        with open("src/pricing-engine/params.json", "r") as f:
            params_input = json.load(f)["VanillaOptionsPricer"]
        self.required_inputs = params_input["required_inputs"]
        self.optional_inputs = params_input["optional_inputs"]
        self.available_models = list(params_input["models"].keys())
        self.h = params_input['h']

    def check_inputs(self) -> None:
        assert np.all(np.isin(self.model, self.available_models)), "Invalid model(s) as input."

    def value(
        self,
        is_call: Optional[np.ndarray[bool]] = None,
        S: Optional[np.ndarray[np.float64]] = None,
        K: Optional[np.ndarray[np.float64]] = None,
        T: Optional[np.ndarray[np.float64]] = None,
        r: Optional[np.ndarray[np.float64]] = None,
        sigma: Optional[np.ndarray[np.float64]] = None,
        b: Optional[np.ndarray[np.float64]] = None,
    ) -> np.ndarray[np.float64]:
        is_call = is_call if is_call is not None else self.is_call
        S = S if S is not None else self.S
        K = K if K is not None else self.K
        T = T if T is not None else self.T
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma
        b = b if b is not None else self.b

        return gbsm.value(is_call, S, K, T, r, sigma, b)
    
    def delta(self) -> np.ndarray[np.float64]:
        up = self.value(self.is_call, self.S + self.h, self.K, self.T, self.r, self.sigma, self.b)
        down = self.value(self.is_call, self.S - self.h, self.K, self.T, self.r, self.sigma, self.b)
        delta = (up - down) / (2 * self.h)
        
        if not np.all((-1.0 <= delta) & (delta <= 1.0)):
            print("[Warning] Delta outside [-1, 1] range. Check inputs or precision.")

        return delta
    
    def gamma(self) -> np.ndarray[np.float64]:
        up = self.value(self.is_call, self.S + self.h, self.K, self.T, self.r, self.sigma, self.b)
        curr = self.value(self.is_call, self.S, self.K, self.T, self.r, self.sigma, self.b)
        down = self.value(self.is_call, self.S - self.h, self.K, self.T, self.r, self.sigma, self.b)
        gamma = (up - 2 * curr + down) / (self.h**2)

        return gamma        

    def vega(self) -> np.ndarray[np.float64]:
        up = self.value(self.is_call, self.S, self.K, self.T, self.r, self.sigma + self.h, self.b)
        down = self.value(self.is_call, self.S, self.K, self.T, self.r, self.sigma - self.h, self.b)
        vega = 0.01 * (up - down) / (2 * self.h)
        return vega
    
    def rho(self) -> np.ndarray[np.float64]:
        up = self.value(self.is_call, self.S, self.K, self.T, self.r + self.h, self.sigma, self.b)
        down = self.value(self.is_call, self.S, self.K, self.T, self.r - self.h, self.sigma, self.b)
        rho = 0.01 * (up - down) / (2 * self.h)
        return rho
    
    def theta(self) -> np.ndarray[np.float64]:
        dt = 1.0 / 365
        curr_day = self.value(self.is_call, self.S, self.K, self.T, self.r, self.sigma, self.b)
        next_day = self.value(self.is_call, self.S, self.K, self.T - dt, self.r, self.sigma, self.b)
        theta = next_day - curr_day
        return theta


if __name__ == "__main__":
    nb_sims = 1_000_000
    models = np.random.choice(["gbsm"], nb_sims)
    flavor = np.random.choice([True, False], nb_sims)
    S = np.random.uniform(80.0, 100.0, nb_sims)
    K = np.random.uniform(80.0, 100.0, nb_sims)
    T = np.random.uniform(0.1, 4.0, nb_sims)
    r = np.random.uniform(0.04, 0.06, nb_sims)
    sigma = np.random.uniform(0.12, 0.65, nb_sims)
    b = np.full(S.shape, 0.0)

    start = time.perf_counter()

    pricer = VanillaOptionsPricer(models, flavor, S, K, T, r, sigma, b)
    value = pricer.value()
    delta = pricer.delta()
    gamma = pricer.gamma()
    rho = pricer.rho()
    theta = pricer.theta()
    vega = pricer.vega()

    end = time.perf_counter()
    elapsed_us = (end - start) * 1e6

    print(value)
    print(delta)
    print(gamma)
    print(vega)
    print(theta)
    print(rho)
    print(f"Elapsed time: {elapsed_us:.2f} µs")