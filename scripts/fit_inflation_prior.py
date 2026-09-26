"""Fit the reconcile stage's claim-regime inflation prior to the gold set.

The spec originally carried ``ancient_source_bias_prior_mu: 0.3`` -- a 1.35x
inflation whose three-sigma ceiling is 6x. The documented cases run 4x-20x
(Gaugamela, Nicaea, the Helvetii, Xerxes), so a prior centred there cannot
represent a single one of them. Worse, for most ancient battles there is no
independent anchor: every surviving figure descends from one classical author,
the likelihood contributes nothing, and the posterior *is* the prior. The
number chosen here is therefore not a modelling nicety, it is the estimate.

So it is measured rather than asserted. ``tests/fixtures/gold/reconcile/``
pairs a source's figure against a named modern scholar's estimate for the same
force, and this script fits the distribution of ``log(source / modern)`` to a
two-component mixture:

    faithful    -- the source had an administrative substrate and lands close
                   to the modern figure (Cannae; the staff-return era)
    rhetorical  -- the figure is a literary or political construct
                   (Herodotus on Persia; the crusade chronicles)

A single Gaussian fits neither, because the two modes are a genuine mixture of
generative stories rather than the tails of one. Per-regime mixture weights
then say how often each regime draws from which component.

Usage::

    python -m scripts.fit_inflation_prior
    python -m scripts.fit_inflation_prior --out config/inflation_priors.yaml
    python -m scripts.fit_inflation_prior --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as stats
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "Case",
    "MixtureFit",
    "fit_era_effect",
    "fit_mixture",
    "load_cases",
    "main",
    "regime_weights",
]

GOLD_DIR: Path = Path("tests/fixtures/gold/reconcile")
DEFAULT_OUT: Path = Path("config/inflation_priors.yaml")

# The anchor. A modern scholarly reconstruction is the scale every other regime
# is measured against, so its bias is pinned to exactly zero rather than
# fitted -- that is what makes an estimate mean "what a modern scholar would
# say" instead of "the average of whatever this corpus happens to hold".
ANCHOR_REGIME = "modern_scholarly"

# EM needs a starting split. These are deliberately crude: the faithful
# component near no inflation, the rhetorical one near a five-fold
# exaggeration, which is roughly where the chronicle cases sit.
_INIT_FAITHFUL_MU = 0.1
_INIT_RHETORICAL_MU = 1.6
_INIT_SD = 0.6
_MIN_SD = 0.15  # keeps a component from collapsing onto a single point
_MAX_ITERATIONS = 500
_TOLERANCE = 1e-9

# Beta(1, 1) pseudo-counts. Some regimes carry four or five cases, and an
# unsmoothed weight of exactly 0.0 or 1.0 would assert more than that supports.
_BETA_PRIOR_A = 1.0
_BETA_PRIOR_B = 1.0

_MIN_CASES_FOR_MIXTURE = 4


@dataclass(frozen=True)
class Case:
    """One gold-set row: a source's figure against a modern estimate."""

    case_id: str
    regime: str
    log_ratio: float
    tradition: str
    quantity: str
    year_astronomical: int | None


@dataclass(frozen=True)
class MixtureFit:
    """A fitted two-component mixture over log inflation ratios."""

    faithful_mu: float
    faithful_sd: float
    rhetorical_mu: float
    rhetorical_sd: float
    weight_rhetorical: float
    n: int
    log_likelihood: float

    def as_dict(self) -> dict[str, float | int]:
        """Render the fit for serialisation.

        Returns:
            The fitted parameters, rounded for a human-readable config file.
        """
        return {
            "faithful_mu": round(self.faithful_mu, 4),
            "faithful_sd": round(self.faithful_sd, 4),
            "rhetorical_mu": round(self.rhetorical_mu, 4),
            "rhetorical_sd": round(self.rhetorical_sd, 4),
            "weight_rhetorical": round(self.weight_rhetorical, 4),
            "n": self.n,
            "log_likelihood": round(self.log_likelihood, 4),
        }


def load_cases(gold_dir: Path = GOLD_DIR) -> list[Case]:
    """Read every gold-set case file.

    Args:
        gold_dir: Directory holding the ``cases_*.jsonl`` files.

    Returns:
        Every row carrying a log ratio, across all traditions.

    Raises:
        SystemExit: If the directory holds no case files. Fitting a prior to
            nothing would produce a confident number backed by no evidence.
    """
    paths = sorted(gold_dir.glob("cases_*.jsonl"))
    if not paths:
        raise SystemExit(
            f"No cases_*.jsonl under {gold_dir}. Build the gold set before fitting a prior."
        )

    cases: list[Case] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ratio = row.get("log_ratio")
            if ratio is None:
                continue
            cases.append(
                Case(
                    case_id=str(row["case_id"]),
                    regime=str(row["ancient_claim_regime"]),
                    log_ratio=float(ratio),
                    tradition=str(row.get("tradition", "unknown")),
                    quantity=str(row.get("quantity", "troops")),
                    year_astronomical=(
                        None
                        if row.get("year_astronomical") is None
                        else int(row["year_astronomical"])
                    ),
                )
            )
    return cases


def _normal_pdf(x: float, mu: float, sd: float) -> float:
    """Density of a normal at x, floored so the E step cannot divide by zero.

    Args:
        x: The point to evaluate at.
        mu: Component mean.
        sd: Component standard deviation.

    Returns:
        The density, never smaller than 1e-300.
    """
    z = (x - mu) / sd
    return max(math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi)), 1e-300)


def fit_mixture(values: list[float]) -> MixtureFit:
    """Fit a two-component normal mixture by expectation-maximisation.

    The components are ordered on exit: the lower mean is "faithful", the
    higher "rhetorical". Ordering by mean rather than by index is what stops
    the label switching that would otherwise make the output depend on where
    EM happened to start.

    Args:
        values: Log inflation ratios.

    Returns:
        The fitted mixture.

    Raises:
        SystemExit: If there are too few points to support two components.
    """
    if len(values) < _MIN_CASES_FOR_MIXTURE:
        raise SystemExit(
            f"Need at least {_MIN_CASES_FOR_MIXTURE} cases to fit two components, "
            f"got {len(values)}"
        )

    mu_a, mu_b = _INIT_FAITHFUL_MU, _INIT_RHETORICAL_MU
    sd_a = sd_b = _INIT_SD
    weight_b = 0.5
    previous = -math.inf
    log_likelihood = previous

    for _ in range(_MAX_ITERATIONS):
        resp: list[float] = []
        log_likelihood = 0.0
        for x in values:
            density_a = (1.0 - weight_b) * _normal_pdf(x, mu_a, sd_a)
            density_b = weight_b * _normal_pdf(x, mu_b, sd_b)
            total = density_a + density_b
            resp.append(density_b / total)
            log_likelihood += math.log(total)

        n_b = sum(resp)
        n_a = len(values) - n_b
        if n_a < 1e-6 or n_b < 1e-6:
            # One component has taken everything; the data do not support two.
            break

        mu_a = sum((1.0 - r) * x for r, x in zip(resp, values, strict=True)) / n_a
        mu_b = sum(r * x for r, x in zip(resp, values, strict=True)) / n_b
        var_a = sum((1.0 - r) * (x - mu_a) ** 2 for r, x in zip(resp, values, strict=True)) / n_a
        var_b = sum(r * (x - mu_b) ** 2 for r, x in zip(resp, values, strict=True)) / n_b
        sd_a = max(math.sqrt(max(var_a, 0.0)), _MIN_SD)
        sd_b = max(math.sqrt(max(var_b, 0.0)), _MIN_SD)
        weight_b = n_b / len(values)

        if abs(log_likelihood - previous) < _TOLERANCE:
            break
        previous = log_likelihood

    if mu_a <= mu_b:
        return MixtureFit(mu_a, sd_a, mu_b, sd_b, weight_b, len(values), log_likelihood)
    return MixtureFit(mu_b, sd_b, mu_a, sd_a, 1.0 - weight_b, len(values), log_likelihood)


def regime_weights(cases: list[Case], fit: MixtureFit) -> dict[str, dict[str, Any]]:
    """Estimate each regime's share of the rhetorical component.

    A regime's weight is the Beta-smoothed mean responsibility of its cases.
    The empirical median is reported alongside so a reader can check the fit
    against something they can compute by hand.

    Args:
        cases: Every gold-set case.
        fit: The pooled mixture responsibilities are computed against.

    Returns:
        Per regime: smoothed weight, case count, median log ratio and the
        equivalent multiplicative factor.
    """
    grouped: dict[str, list[float]] = {}
    for case in cases:
        grouped.setdefault(case.regime, []).append(case.log_ratio)

    out: dict[str, dict[str, Any]] = {}
    for regime, values in sorted(grouped.items()):
        responsibilities = []
        for x in values:
            density_a = (1.0 - fit.weight_rhetorical) * _normal_pdf(
                x, fit.faithful_mu, fit.faithful_sd
            )
            density_b = fit.weight_rhetorical * _normal_pdf(
                x, fit.rhetorical_mu, fit.rhetorical_sd
            )
            responsibilities.append(density_b / (density_a + density_b))

        smoothed = (sum(responsibilities) + _BETA_PRIOR_A) / (
            len(values) + _BETA_PRIOR_A + _BETA_PRIOR_B
        )
        out[regime] = {
            "weight_rhetorical": round(smoothed, 4),
            "n": len(values),
            "median_log_ratio": round(stats.median(values), 4),
            "median_factor": round(math.exp(stats.median(values)), 3),
        }
    return out



def fit_era_effect(cases: list[Case], ancient_cutoff_year: int = 500) -> tuple[float, float]:
    """Estimate how much more ancient-battle figures inflate than later ones.

    The model needs this as a prior on its era fallback term, which carries
    most of the inflation signal because only about 2% of real reports name
    their tradition. It must be *measured*: taking the gap between the
    mixture's two components instead gives +1.58, while the gold set gives
    roughly -0.45, so the sign itself would be wrong.

    The estimate is deliberately imprecise. The split is confounded with claim
    regime, since chronicle cases cluster after the cutoff, and the gold set is
    selection-biased toward notorious exaggerations. The standard error is
    therefore widened rather than reported at face value.

    Args:
        cases: Every gold-set case.
        ancient_cutoff_year: The astronomical year below which a battle counts
            as ancient.

    Returns:
        ``(mu, sd)`` for a normal prior on the ancient-minus-later contrast.
    """
    ancient = [c.log_ratio for c in cases
               if c.year_astronomical is not None and c.year_astronomical < ancient_cutoff_year]
    later = [c.log_ratio for c in cases
             if c.year_astronomical is not None and c.year_astronomical >= ancient_cutoff_year]

    if len(ancient) < 2 or len(later) < 2:
        return 0.0, 0.5

    mu = stats.mean(ancient) - stats.mean(later)
    se = math.sqrt(
        stats.variance(ancient) / len(ancient) + stats.variance(later) / len(later)
    )
    # Doubled: the contrast is confounded with regime and the sample is not a
    # random draw from the corpus, so the sampling error understates how much
    # this could be wrong.
    return mu, max(2.0 * se, 0.3)


def _digest(gold_dir: Path) -> str:
    """Hash the gold-set inputs so a fitted prior is traceable to them.

    Args:
        gold_dir: Directory holding the case files.

    Returns:
        A short hex digest over every case file's name and contents.
    """
    sha = hashlib.sha256()
    for path in sorted(gold_dir.glob("cases_*.jsonl")):
        sha.update(path.name.encode("utf-8"))
        sha.update(path.read_bytes())
    return sha.hexdigest()[:16]


def _render_yaml(
    fit: MixtureFit,
    weights: dict[str, dict[str, Any]],
    digest: str,
    n: int,
    era: tuple[float, float],
) -> str:
    """Render the fitted prior as the YAML the reconcile spec reads.

    Args:
        fit: The pooled mixture.
        weights: Per-regime weights from :func:`regime_weights`.
        digest: The gold-set digest.
        n: Number of cases fitted.
        era: ``(mu, sd)`` for the ancient-minus-later contrast.

    Returns:
        The file's full text.
    """
    lines = [
        "# Fitted by scripts/fit_inflation_prior.py -- do not hand-edit.",
        "#",
        "# The distribution of log(source figure / modern scholarly estimate)",
        "# over tests/fixtures/gold/reconcile/. Two components, because Cannae",
        "# (accepted at face value) and Nicaea (10-20x) are different",
        "# generative stories rather than two tails of one.",
        "#",
        f"# Fitted {date.today().isoformat()} from {n} cases.",
        f"# Gold-set digest: {digest}",
        "",
        "provenance:",
        f"  fitted_on: {date.today().isoformat()}",
        f"  gold_set_digest: {digest}",
        f"  n_cases: {n}",
        "  script: scripts/fit_inflation_prior.py",
        "",
        "# The anchor regime is pinned to exactly zero and is never fitted.",
        f"anchor_regime: {ANCHOR_REGIME}",
        "",
        "mixture:",
    ]
    lines += [f"  {key}: {value}" for key, value in fit.as_dict().items()]
    lines += [
        "",
        "# Prior on the era fallback: how much more an ancient battle's figures",
        "# inflate than a later one's. Measured, because the mixture-component",
        "# gap that was used before is a different quantity and has the",
        "# opposite sign. Widened; see fit_era_effect.",
        "era:",
        f"  mu: {round(era[0], 4)}",
        f"  sd: {round(era[1], 4)}",
        "",
        "regimes:",
    ]
    for regime, info in weights.items():
        lines.append(f"  {regime}:")
        lines += [f"    {key}: {value}" for key, value in info.items()]
    return "\n".join(lines) + "\n"


def _print_summary(cases: list[Case], fit: MixtureFit, weights: dict[str, dict[str, Any]]) -> None:
    """Print the fit as a table.

    Args:
        cases: Every gold-set case.
        fit: The pooled mixture.
        weights: Per-regime weights.
    """
    values = [case.log_ratio for case in cases]
    median = stats.median(values)
    print(f"{len(cases)} cases")
    print(f"  pooled median  {median:+.3f}  ({math.exp(median):.2f}x)")
    print()
    print(f"  {'component':11s} {'mu':>7s} {'sd':>7s} {'share':>7s}   factor")
    print(
        f"  {'faithful':11s} {fit.faithful_mu:+7.3f} {fit.faithful_sd:7.3f} "
        f"{1.0 - fit.weight_rhetorical:7.2f}   {math.exp(fit.faithful_mu):.2f}x"
    )
    print(
        f"  {'rhetorical':11s} {fit.rhetorical_mu:+7.3f} {fit.rhetorical_sd:7.3f} "
        f"{fit.weight_rhetorical:7.2f}   {math.exp(fit.rhetorical_mu):.2f}x"
    )
    print()
    print(f"  {'regime':26s} {'n':>3s} {'w_rhet':>7s} {'median':>8s} {'factor':>7s}")
    for regime, info in weights.items():
        print(
            f"  {regime:26s} {info['n']:3d} {info['weight_rhetorical']:7.3f} "
            f"{info['median_log_ratio']:+8.3f} {info['median_factor']:6.2f}x"
        )


def main(argv: list[str] | None = None) -> None:
    """Fit the prior and write it, or print it.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description="Fit the claim-regime inflation prior.")
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=GOLD_DIR,
        metavar="DIR",
        help=f"Gold-set directory (default: {GOLD_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        metavar="PATH",
        help=f"Where to write the fitted prior (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fit and the YAML without writing anything",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.gold_dir)
    fit = fit_mixture([case.log_ratio for case in cases])
    weights = regime_weights(cases, fit)
    digest = _digest(args.gold_dir)
    era = fit_era_effect(cases)

    _print_summary(cases, fit, weights)
    print()
    print(f"  era contrast (ancient minus later): {era[0]:+.3f} (sd {era[1]:.3f})")

    text = _render_yaml(fit, weights, digest, len(cases), era)
    if args.dry_run:
        print()
        print(text)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
