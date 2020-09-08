from typing import Callable, Iterable, Iterator, Optional, Tuple, Any, Dict

import numpy as np
from scipy.optimize import OptimizeResult, root
from scipy.sparse.linalg import LinearOperator


def natural_jfnk(
    fun: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    w: np.ndarray,
    milestones: Iterable[float],
    preconditioner: Optional[Callable[[float, np.ndarray], LinearOperator]] = None,
    root_options: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[float, OptimizeResult]]:
    """Generate pairs (mu, x) such that `x` is a root of fun(`x`)

    and the mu contain the `milestones`.

    `root_options` are passed on to root; e.g. maxiter, tol_norm.
    """

    options = {"disp": True, "maxiter": 9, "jac_options": {}, **(root_options or {})}

    mu = d_mu = np.NINF
    for milestone in milestones:
        if np.isneginf(mu):
            mu = milestone
            if preconditioner:
                options["jac_options"].update(inner_M=preconditioner(mu, w))
            sol = root(fun(mu), w, method="krylov", options=options)
            assert sol.success
            yield mu, sol
            continue

        if np.isneginf(d_mu):
            d_mu = milestone - mu

        while mu < milestone:

            mu = min(mu + d_mu, milestone)
            if preconditioner:
                options["jac_options"].update(inner_M=preconditioner(mu, w))

            sol = root(fun(mu), sol.x, method="krylov", options=options)
            if not sol.success:
                print(f"No convergence for mu={mu}.", sol.message)
                mu -= d_mu
                d_mu /= 2
                continue

            d_mu *= np.sqrt(options["maxiter"] / sol.nit)
            yield mu, sol
