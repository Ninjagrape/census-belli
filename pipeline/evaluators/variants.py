"""Sensitivity variants: declarative specs and the pure transforms they apply.

Each variant in ``agents/evaluate.yaml`` is a mapping ``{name, kind,
description, ...options}``. The transforms here never mutate their inputs;
the stage refits the model on the transformed sides or design and compares
rankings. :func:`designs_equivalent` lets it report a variant as a no-op
without refitting.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args

import numpy as np

from pipeline.imputers.likelihood import NeutralDraws
from pipeline.imputers.priors import era_of_year
from pipeline.modelers.load import ImputedRound
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import CommanderInput, ModelDesign, SideInput

__all__ = [
    "NotEvaluableError",
    "VariantKind",
    "VariantSpec",
    "apply_battle_filter",
    "apply_side_transform",
    "collapse_three_level",
    "designs_equivalent",
    "fill_weak_slopes",
    "observed_only_x_diff",
    "parse_variants",
    "read_imputed_side_ids",
    "g_unlabelled_sd",
    "g_unlabelled_slopes",
    "prior_overrides",
    "shift_rounds",
]

VariantKind = Literal[
    "side_transform",
    "battle_filter",
    "design_transform",
    "x_source",
    "priors",
    "post_hoc",
    "regime_shift",
]

EQUAL_METHOD: Final[str] = "variant_equal"
_DEFAULT_FLAT_PRIORS: Final[Mapping[str, float]] = {"skill_prior_sd": 5.0, "beta_force_sd": 2.0}

# name -> (kind, allowed option keys, defaults)
_KNOWN: Final[Mapping[str, tuple[str, frozenset[str], Mapping[str, Any]]]] = {
    "equal_attribution": ("side_transform", frozenset(), {}),
    "field_commanders_only": (
        "side_transform",
        frozenset({"exclude_roles"}),
        {"exclude_roles": ["sovereign", "nominal"]},
    ),
    "drop_low_quality": ("battle_filter", frozenset({"min_quality"}), {"min_quality": 0.3}),
    "exclude_extreme_ratio": (
        "battle_filter",
        frozenset({"max_abs_log_ratio"}),
        {"max_abs_log_ratio": 2.0},
    ),
    "three_level_outcome": ("design_transform", frozenset(), {}),
    "observed_only": ("x_source", frozenset(), {}),
    "weak_hyperpriors": ("priors", frozenset(), {}),
    "replacement_50th": (
        "post_hoc",
        frozenset({"replacement_percentile"}),
        {"replacement_percentile": 50},
    ),
    "realised_war": ("post_hoc", frozenset({"rank_by"}), {"rank_by": "realised"}),
    "g_unlabelled_plus_1sd": ("regime_shift", frozenset({"n_sd"}), {"n_sd": 1.0}),
    "g_unlabelled_minus_1sd": ("regime_shift", frozenset({"n_sd"}), {"n_sd": -1.0}),
}
_KINDS: Final[frozenset[str]] = frozenset(get_args(VariantKind))
_PRIOR_FIELDS: Final[frozenset[str]] = frozenset(f.name for f in dataclasses.fields(ModelPriors))


class NotEvaluableError(Exception):
    """A variant cannot be run on the inputs available (reported, not failed)."""


@dataclass(frozen=True)
class VariantSpec:
    """One declared sensitivity variant.

    Attributes:
        name: Unique variant name.
        kind: Which part of the pipeline the variant changes.
        description: Free text for the report.
        options: Validated options with defaults filled in.
    """

    name: str
    kind: VariantKind
    description: str
    options: Mapping[str, Any]


def _number(name: str, options: Mapping[str, Any], key: str) -> float:
    value = options[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"variant {name!r}: option {key!r} must be a finite number")
    return float(value)


def _check_values(name: str, options: Mapping[str, Any]) -> None:
    if "exclude_roles" in options:
        roles = options["exclude_roles"]
        if not isinstance(roles, list | tuple) or not all(isinstance(r, str) for r in roles):
            raise ValueError(f"variant {name!r}: exclude_roles must be a list of strings")
    if "min_quality" in options and not 0.0 <= _number(name, options, "min_quality") <= 1.0:
        raise ValueError(f"variant {name!r}: min_quality must be in [0, 1]")
    if "max_abs_log_ratio" in options and _number(name, options, "max_abs_log_ratio") <= 0.0:
        raise ValueError(f"variant {name!r}: max_abs_log_ratio must be positive")
    if (
        "replacement_percentile" in options
        and not 0.0 < _number(name, options, "replacement_percentile") < 100.0
    ):
        raise ValueError(f"variant {name!r}: replacement_percentile must be in (0, 100)")
    if "n_sd" in options and _number(name, options, "n_sd") == 0.0:
        raise ValueError(f"variant {name!r}: n_sd must be non-zero")
    if "rank_by" in options and options["rank_by"] != "realised":
        raise ValueError(f"variant {name!r}: rank_by must be 'realised'")


def _validate_priors(name: str, options: dict[str, Any]) -> dict[str, Any]:
    extra = set(options) - _PRIOR_FIELDS
    if extra:
        raise ValueError(f"variant {name!r}: not ModelPriors field(s) {sorted(extra)}")
    merged = dict(options) if options else dict(_DEFAULT_FLAT_PRIORS)
    for key in merged:
        _number(name, merged, key)
    return merged


def _validate_unknown(name: str, kind: str, options: dict[str, Any]) -> dict[str, Any]:
    """A new name is allowed only when its options say what it does."""
    if kind == "priors" and options:
        return _validate_priors(name, options)
    sufficient = (kind == "side_transform" and "exclude_roles" in options) or (
        kind == "battle_filter" and ("min_quality" in options or "max_abs_log_ratio" in options)
    )
    if not sufficient:
        raise ValueError(
            f"variant {name!r} of kind {kind!r} is not a known variant and its options "
            "do not define one"
        )
    _check_values(name, options)
    return dict(options)


def _validate_options(name: str, kind: str, options: dict[str, Any]) -> dict[str, Any]:
    """Check options against the variant's contract and fill defaults."""
    known = _KNOWN.get(name)
    if known is None:
        return _validate_unknown(name, kind, options)
    expected_kind, allowed, defaults = known
    if kind != expected_kind:
        raise ValueError(f"variant {name!r} has kind {expected_kind!r}, not {kind!r}")
    if kind == "priors":
        return _validate_priors(name, options)
    extra = set(options) - allowed
    if extra:
        raise ValueError(f"variant {name!r}: unknown option(s) {sorted(extra)}")
    merged = {**defaults, **options}
    _check_values(name, merged)
    return merged


def parse_variants(raw: Sequence[Mapping[str, Any]]) -> list[VariantSpec]:
    """Parse and validate the variant list from ``agents/evaluate.yaml``.

    Args:
        raw: Mappings with ``name``, ``kind``, ``description`` and options.

    Returns:
        One spec per entry, in order.

    Raises:
        ValueError: On a missing name, unknown kind, duplicate name, or
            options that do not fit the variant.
    """
    specs: list[VariantSpec] = []
    seen: set[str] = set()
    for entry in raw:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"variant entry has no name: {dict(entry)!r}")
        if name in seen:
            raise ValueError(f"duplicate variant name {name!r}")
        kind = entry.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"variant {name!r} has unknown kind {kind!r}")
        seen.add(name)
        options = {k: v for k, v in entry.items() if k not in ("name", "kind", "description")}
        specs.append(
            VariantSpec(
                name=name,
                kind=kind,
                description=str(entry.get("description", "")),
                options=_validate_options(name, kind, options),
            )
        )
    return specs


def _require_kind(spec: VariantSpec, kind: str) -> None:
    if spec.kind != kind:
        raise ValueError(f"variant {spec.name!r} is {spec.kind!r}, expected {kind!r}")


def _equal_side(side: SideInput) -> SideInput:
    first_row: dict[int, CommanderInput] = {}
    for c in side.commanders:
        first_row.setdefault(c.general_id, c)
    if not first_row:
        return side
    weight = 1.0 / len(first_row)
    commanders = tuple(
        dataclasses.replace(c, attribution_weight=weight, attribution_method=EQUAL_METHOD)
        for c in first_row.values()
    )
    return dataclasses.replace(side, commanders=commanders)


def _has_positive_weight(commanders: Sequence[CommanderInput]) -> bool:
    return any(math.isfinite(c.attribution_weight) and c.attribution_weight > 0 for c in commanders)


def apply_side_transform(
    spec: VariantSpec, sides: Sequence[SideInput]
) -> tuple[list[SideInput], dict[str, int]]:
    """Apply a ``side_transform`` variant to every side.

    Args:
        spec: The variant.
        sides: Input sides, left untouched.

    Returns:
        ``(new_sides, counts)``; ``counts`` has ``sides_changed``,
        ``commanders_dropped``, ``sides_kept_unchanged`` (role filtering only:
        sides where every commander was excluded and so none were) and
        ``sides_reweighted`` (role filtering only: sides whose kept commanders
        had no positive finite weight, so they were given equal weights).
        Role filtering keeps a commander whose ``command_role`` is None.
    """
    _require_kind(spec, "side_transform")
    roles = set(spec.options["exclude_roles"]) if "exclude_roles" in spec.options else None
    counts = {
        "sides_changed": 0,
        "commanders_dropped": 0,
        "sides_kept_unchanged": 0,
        "sides_reweighted": 0,
    }
    out: list[SideInput] = []
    for side in sides:
        if roles is None:
            new = _equal_side(side)
        else:
            # A command_role of None (the loader did not read one) is never in
            # `roles`, so that commander is kept.
            kept = tuple(c for c in side.commanders if c.command_role not in roles)
            if side.commanders and not kept:
                counts["sides_kept_unchanged"] += 1
                new = side
            else:
                counts["commanders_dropped"] += len(side.commanders) - len(kept)
                new = dataclasses.replace(side, commanders=kept)
                if kept and not _has_positive_weight(kept):
                    # Weights are renormalised per side, so all-zero weights
                    # would give the side no credit at all.
                    new = _equal_side(new)
                    counts["sides_reweighted"] += 1
        if new.commanders != side.commanders:
            counts["sides_changed"] += 1
        out.append(new)
    return out, counts


def apply_battle_filter(
    spec: VariantSpec,
    sides: Sequence[SideInput],
    *,
    battle_quality: Mapping[int, float] | None = None,
    measured_x_diff: Mapping[int, float] | None = None,
) -> tuple[list[SideInput], dict[str, int]]:
    """Drop every side of the battles a ``battle_filter`` variant rejects.

    Battles missing from the relevant mapping are kept, since an absent score
    is not evidence of a bad battle.

    Args:
        spec: The variant.
        sides: Input sides.
        battle_quality: ``battle_id`` to data quality score.
        measured_x_diff: ``battle_id`` to the measurement-only log force ratio
            (see :func:`observed_only_x_diff`); imputed values are outcome-
            conditioned and must not be used here.

    Returns:
        ``(kept_sides, counts)`` with ``battles_dropped`` and ``sides_dropped``.

    Raises:
        ValueError: If the mapping the variant needs was not supplied.
    """
    _require_kind(spec, "battle_filter")
    if "min_quality" in spec.options:
        if battle_quality is None:
            raise ValueError(f"variant {spec.name!r} needs battle_quality")
        floor = float(spec.options["min_quality"])
        rejected = {b for b, q in battle_quality.items() if q < floor}
    else:
        if measured_x_diff is None:
            raise ValueError(f"variant {spec.name!r} needs measured_x_diff")
        cutoff = float(spec.options["max_abs_log_ratio"])
        rejected = {b for b, x in measured_x_diff.items() if abs(x) > cutoff}
    kept = [s for s in sides if s.battle_id not in rejected]
    present = {s.battle_id for s in sides}
    return kept, {
        "battles_dropped": len(rejected & present),
        "sides_dropped": len(sides) - len(kept),
    }


def collapse_three_level(design: ModelDesign) -> ModelDesign:
    """Merge outcome levels 0 and 1, and 3 and 4, by interval censoring.

    Args:
        design: The five-level design.

    Returns:
        A copy whose bounds are widened to the merged levels; ``y_a`` is
        unchanged since the win / draw / loss class is preserved.
    """
    k_lo = np.where(design.k_lo <= 1, 0, np.where(design.k_lo >= 3, 3, 2)).astype(np.int64)
    k_hi = np.where(design.k_hi <= 1, 1, np.where(design.k_hi >= 3, 4, 2)).astype(np.int64)
    return dataclasses.replace(design, k_lo=k_lo, k_hi=k_hi)


def observed_only_x_diff(
    design: ModelDesign, obs_mean: np.ndarray, obs_sd: np.ndarray
) -> dict[int, float]:
    """Fixed log force ratios for battles where both sides were measured.

    Args:
        design: The design.
        obs_mean: Observed log-troop mean per side index, NaN if unmeasured.
        obs_sd: Observed log-troop sd per side index, NaN if unmeasured.

    Returns:
        ``battle_id`` to ``obs_mean[A] - obs_mean[B]``.
    """
    observed = np.isfinite(obs_mean) & np.isfinite(obs_sd) & (obs_sd > 0)
    both = observed[design.side_a] & observed[design.side_b]
    diff = obs_mean[design.side_a] - obs_mean[design.side_b]
    return {int(b): float(d) for b, d, ok in zip(design.battle_ids, diff, both, strict=True) if ok}


def prior_overrides(priors: ModelPriors, options: Mapping[str, Any]) -> ModelPriors:
    """Return priors with the given fields replaced.

    Args:
        priors: Base priors.
        options: Field name to new value.

    Returns:
        New priors.

    Raises:
        ValueError: If a key is not a ``ModelPriors`` field.
    """
    extra = set(options) - _PRIOR_FIELDS
    if extra:
        raise ValueError(f"not ModelPriors field(s): {sorted(extra)}")
    return dataclasses.replace(priors, **{k: float(v) for k, v in options.items()})


_EXACT_FIELDS: Final[tuple[str, ...]] = (
    "battle_ids",
    "side_a",
    "side_b",
    "k_lo",
    "k_hi",
    "side_ids",
    "force_group",
    "defender_diff",
    "side_kind",
    "cmd_side",
    "cmd_general",
    "general_ids",
)

_CLOSE_FIELDS: Final[tuple[str, ...]] = (
    "cmd_weight",
    "side_unknown_share",
    "side_unattributed_share",
)


def designs_equivalent(a: ModelDesign, b: ModelDesign) -> bool:
    """Whether two designs would give the same fit.

    Args:
        a: First design.
        b: Second design.

    Returns:
        True when battles, sides, outcome bounds, commander entries and
        general ids match and the weights and latent shares agree to
        ``np.allclose``.
    """
    if any(not np.array_equal(getattr(a, f), getattr(b, f)) for f in _EXACT_FIELDS):
        return False
    # The latent shares change a fit just as the commander weights do: two designs
    # with the same named weights but a different anonymous remainder differ.
    return all(
        getattr(a, f).shape == getattr(b, f).shape
        and bool(np.allclose(getattr(a, f), getattr(b, f)))
        for f in _CLOSE_FIELDS
    )


def g_unlabelled_sd(neutral: NeutralDraws) -> float:
    """Posterior sd of reconcile's shared unlabelled-regime offset.

    Args:
        neutral: Reconcile's neutral draws.

    Returns:
        ``std(g_unlabelled, ddof=1)``.

    Raises:
        NotEvaluableError: If the file predates the regime export, has fewer
            than two draws, or the offset has zero spread.
    """
    g = neutral.g_unlabelled
    if g is None:
        raise NotEvaluableError(
            "the neutral draws file has no g_unlabelled (written before reconcile exported "
            "the regime draws); re-run reconcile"
        )
    if g.shape[0] < 2:
        raise NotEvaluableError("g_unlabelled has fewer than two draws")
    sd = float(np.std(g, ddof=1))
    if not math.isfinite(sd) or sd <= 0.0:
        raise NotEvaluableError(f"g_unlabelled has zero spread (sd={sd})")
    return sd


def g_unlabelled_slopes(neutral: NeutralDraws) -> dict[int, float]:
    """Slope of each side's neutral draws on ``g_unlabelled`` across draws.

    One least-squares fit per side, vectorised, with an intercept and, when the
    file carries it, ``a_era`` as a second regressor so the era offset is not
    attributed to ``g_unlabelled``. Only the ``g_unlabelled`` coefficient is kept.

    Args:
        neutral: Reconcile's neutral draws; ``g_unlabelled`` must be present.

    Returns:
        ``side_id`` to slope B_s (log troops per unit of ``g_unlabelled``).

    Raises:
        NotEvaluableError: If ``g_unlabelled`` is absent.
    """
    g = neutral.g_unlabelled
    if g is None:
        raise NotEvaluableError("the neutral draws file has no g_unlabelled")
    columns = [np.ones_like(g), g]
    if neutral.a_era is not None:
        columns.append(neutral.a_era)
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, neutral.draws, rcond=None)
    return {int(s): float(b) for s, b in zip(neutral.side_id, coef[1], strict=True)}


def read_imputed_side_ids(root: Path, scale: str, indices: Sequence[int]) -> frozenset[int]:
    """Side ids flagged ``is_imputed`` in any of the given rounds' ``sides.csv``.

    Args:
        root: The imputed-data root.
        scale: The scale directory.
        indices: Round numbers to read.

    Returns:
        The flagged side ids.

    Raises:
        NotEvaluableError: If a round's file is unreadable or lacks the column.
    """
    flagged: set[int] = set()
    for n in indices:
        path = root / scale / f"round_{n}" / "sides.csv"
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["is_imputed"].strip().lower() == "true":
                        flagged.add(int(row["side_id"]))
        except (OSError, KeyError, ValueError) as exc:
            raise NotEvaluableError(f"cannot read is_imputed from {path}: {exc!r}") from exc
    return frozenset(flagged)


def fill_weak_slopes(
    slopes: Mapping[int, float],
    neutral: NeutralDraws,
    sides: Sequence[SideInput],
    imputed: frozenset[int],
    era_bins: tuple[tuple[str, int | None], ...],
) -> tuple[dict[int, float], dict[str, int]]:
    """Give every side without its own usable slope its era's observed slope.

    A side has no usable slope when it is missing from the neutral draws, or
    impute filled it from the prior (``imputed``): its slope on
    ``g_unlabelled`` then measures nothing about the side itself. Such a side
    takes the precision-weighted mean slope of the observed sides in the same
    era, precision being 1 / variance of the side's neutral draws. An era with
    no observed side falls back to the weighted mean over all observed sides,
    and to 0 only when there is none at all.

    Args:
        slopes: Per-side slopes from :func:`g_unlabelled_slopes`.
        neutral: The draws the slopes came from (for the precisions).
        sides: Every side the rounds cover, with its battle year.
        imputed: Side ids flagged ``is_imputed`` in the imputed rounds.
        era_bins: Impute's era bins, so eras match the imputation model's.

    Returns:
        ``(slopes, counts)``: the slopes for observed sides as given plus the
        filled ones, and how many sides were filled from an era, from the
        overall mean, or left at 0.
    """
    precision: dict[int, float] = {}
    if neutral.draws.shape[0] > 1:
        variance = np.var(neutral.draws, axis=0, ddof=1)
        for sid, var in zip(neutral.side_id.tolist(), variance.tolist(), strict=True):
            if math.isfinite(var) and var > 0.0:
                precision[int(sid)] = 1.0 / var
    era_of = {s.side_id: era_of_year(s.year, era_bins) for s in sides}
    observed = [
        sid
        for sid, slope in slopes.items()
        if sid not in imputed and sid in precision and math.isfinite(slope)
    ]
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for sid in observed:
        era = era_of.get(sid)
        if era is None:
            continue
        num[era] = num.get(era, 0.0) + precision[sid] * slopes[sid]
        den[era] = den.get(era, 0.0) + precision[sid]
    total_den = sum(den.values())
    overall = sum(num.values()) / total_den if total_den > 0.0 else None
    out = {sid: slopes[sid] for sid in observed}
    counts = {"filled_from_era": 0, "filled_from_overall": 0, "left_at_zero": 0}
    for side in sides:
        if side.side_id in out:
            continue
        era = era_of[side.side_id]
        if den.get(era, 0.0) > 0.0:
            out[side.side_id] = num[era] / den[era]
            counts["filled_from_era"] += 1
        elif overall is not None:
            out[side.side_id] = overall
            counts["filled_from_overall"] += 1
        else:
            out[side.side_id] = 0.0
            counts["left_at_zero"] += 1
    return out, counts


def shift_rounds(
    rounds: Sequence[ImputedRound], slopes: Mapping[int, float], shift: float
) -> list[ImputedRound]:
    """Shift each side's log troops in every round by ``slopes[side] * shift``.

    Args:
        rounds: Imputed rounds, left untouched.
        slopes: ``side_id`` to B_s; a side absent from it shifts by 0.
        shift: The offset move, ``n_sd * sd(g_unlabelled)``.

    Returns:
        New rounds with the same indices and provenance.
    """
    return [
        dataclasses.replace(
            rnd,
            log_troops={s: v + slopes.get(s, 0.0) * shift for s, v in rnd.log_troops.items()},
        )
        for rnd in rounds
    ]
