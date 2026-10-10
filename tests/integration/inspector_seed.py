"""Synthetic dataset for the inspector tests and for a manual dashboard launch.

``seed_database`` fills a freshly applied schema with three battles (one BC),
six sides, commanders, reports from two sources, reconcile estimates including
the neutral family, a model run, imputed ``missing_data_log`` rows, an LLM call
and crawl rows. ``write_data_root`` builds the matching file tree.

Run ``python -m tests.integration.inspector_seed DATA_ROOT`` to (re)seed
``DATABASE_URL`` and write the files, then point the dashboard at it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine, text

from pipeline.db import apply_schema, get_engine

__all__ = ["BATTLES", "seed_database", "write_data_root"]

# (name, date literal, battle_type)
BATTLES = [
    ("Battle of Marathon", "0490-09-12 BC", "field"),
    ("Battle of Actium", "0031-09-02 BC", "naval"),
    ("Battle of Waterloo", "1815-06-18", "field"),
]
RECONCILE_RUN = "reconcile"
IMPUTATION_RUN = "imputation"
N_ROUNDS = 3


def _scalar(conn: Any, sql: str, params: dict[str, Any]) -> int:
    """Run an INSERT ... RETURNING and return the id."""
    return int(conn.execute(text(sql), params).scalar_one())


def _seed_sources_and_runs(conn: Any) -> dict[str, int]:
    """Insert sources and model runs; return their ids by key."""
    ids: dict[str, int] = {}
    runs = {
        "rec": (RECONCILE_RUN, {"rhat_max": 1.01, "n_divergences": 0}),
        "imp": (IMPUTATION_RUN, {"rhat_max": 1.0}),
    }
    for key, (model_type, diagnostics) in runs.items():
        ids[key] = _scalar(
            conn,
            "INSERT INTO model_runs (run_name, model_type, config, started_at, completed_at, "
            "diagnostics) VALUES (:n, :t, CAST('{}' AS JSONB), now(), now(), "
            "CAST(:d AS JSONB)) RETURNING run_id",
            {"n": f"seed_{key}", "t": model_type, "d": json.dumps(diagnostics)},
        )
    sources = {
        "herodotus": ("Herodotus, Histories", "primary_source", -0.8, 0.3, 1.1, 2),
        "wiki": ("Wikipedia infobox", "wikipedia_infobox", 0.1, 0.2, 0.4, 5),
    }
    for key, (title, stype, mu, sd, sigma, n) in sources.items():
        ids[key] = _scalar(
            conn,
            "INSERT INTO sources (source_type, title, troop_bias_mu, troop_bias_sd, "
            "troop_sigma_mu, bias_n_troop_reports, bias_run_id) "
            "VALUES (CAST(:st AS source_type), :t, :mu, :sd, :sig, :n, :run) RETURNING source_id",
            {"st": stype, "t": title, "mu": mu, "sd": sd, "sig": sigma, "n": n, "run": ids["rec"]},
        )
    return ids


def _seed_battles(conn: Any, ids: dict[str, int]) -> None:
    """Insert battles, sides and the reconcile estimates."""
    for i, (name, date, kind) in enumerate(BATTLES):
        bid = _scalar(
            conn,
            "INSERT INTO battles (name, date_start, battle_type) "
            "VALUES (:n, CAST(:d AS DATE), CAST(:k AS battle_type)) RETURNING battle_id",
            {"n": name, "d": date, "k": kind},
        )
        ids[f"b{i}"] = bid
        for j, label in enumerate(("Side A", "Side B")):
            base = 10_000.0 * (i + 1) * (j + 1)
            ids[f"s{i}{j}"] = _scalar(
                conn,
                "INSERT INTO battle_sides (battle_id, side_label, is_defender, "
                "est_troops_total, est_troops_total_lo, est_troops_total_hi, "
                "est_troops_neutral, est_troops_neutral_lo, est_troops_neutral_hi, "
                "est_troops_neutral_log_mean, est_troops_neutral_log_sd, "
                "est_troops_method, est_troops_n_reports, est_troops_n_sources, "
                "est_troops_run_id, est_troops_neutral_run_id, est_casualties, "
                "est_casualties_lo, est_casualties_hi, est_casualties_method) "
                "VALUES (:b, :l, :d, :t, :tl, :th, :n, :nl, :nh, :lm, 0.3, "
                "'source_disagreement', 2, 2, :run, :run, :c, :cl, :ch, "
                "'single_report_debiased') RETURNING side_id",
                {
                    "b": bid,
                    "l": label,
                    "d": j == 1,
                    "t": base,
                    "tl": base * 0.6,
                    "th": base * 1.7,
                    "n": base * 1.1,
                    "nl": base * 0.7,
                    "nh": base * 1.6,
                    "lm": float(np.log(base * 1.1)),
                    "run": ids["rec"],
                    "c": base * 0.1,
                    "cl": base * 0.05,
                    "ch": base * 0.2,
                },
            )


def _seed_people_and_reports(conn: Any, ids: dict[str, int]) -> None:
    """Insert generals, commanders, reports, casualty reports."""
    people = [("Miltiades", "Q1"), ("Octavian", "Q2"), ("Wellington", None)]
    for k, (name, qid) in enumerate(people):
        gid = _scalar(
            conn,
            "INSERT INTO generals (canonical_name, wikidata_id) VALUES (:n, :q) "
            "RETURNING general_id",
            {"n": name, "q": qid},
        )
        conn.execute(
            text("INSERT INTO general_aliases (general_id, alias_name) VALUES (:g, :a)"),
            {"g": gid, "a": f"{name} the Elder"},
        )
        roles = ("field_commander", "supreme_commander", "field_commander")
        conn.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id, command_role, "
                "attribution_weight, attribution_method, confidence, role_evidence) "
                "VALUES (:b, :s, :g, CAST(:r AS command_role), 1.0, 'role_heuristic', :c, :e)"
            ),
            {
                "b": ids[f"b{k}"],
                "s": ids[f"s{k}0"],
                "g": gid,
                "r": roles[k],
                "c": 0.9 if k < 2 else 0.3,
                "e": f"{name} commanded the army",
            },
        )
    for k in range(len(BATTLES)):
        for src, value, scope in (("herodotus", 12_000.0, "engaged"), ("wiki", 9_000.0, "unknown")):
            conn.execute(
                text(
                    "INSERT INTO troop_reports (side_id, source_id, reported_value, scope, "
                    "extraction_method, extracted_context) "
                    "VALUES (:s, :src, :v, :sc, 'infobox_parser', 'about this many')"
                ),
                {"s": ids[f"s{k}0"], "src": ids[src], "v": value * (k + 1), "sc": scope},
            )
    conn.execute(
        text(
            "INSERT INTO casualty_reports (side_id, source_id, casualty_type, reported_value) "
            "VALUES (:s, :src, 'killed', 192)"
        ),
        {"s": ids["s00"], "src": ids["herodotus"]},
    )


def _seed_logs(conn: Any, ids: dict[str, int]) -> None:
    """Insert missing-data, LLM and crawl rows."""
    for field, cls, imputed in (("troop_total", "mnar", True), ("outcome", "mcar", False)):
        conn.execute(
            text(
                "INSERT INTO missing_data_log (battle_id, side_id, field_name, "
                "missingness_class, was_imputed, imputation_method, imputed_value, "
                "imputed_lo, imputed_hi, imputation_round, notes) "
                "VALUES (:b, :s, :f, CAST(:c AS missingness_class), :w, :m, :v, :lo, :hi, 0, 'n')"
            ),
            {
                "b": ids["b2"],
                "s": ids["s21"],
                "f": field,
                "c": cls,
                "w": imputed,
                "m": "era_polity_prior" if imputed else None,
                "v": 8.5 if imputed else None,
                "lo": 7.9 if imputed else None,
                "hi": 9.1 if imputed else None,
            },
        )
    conn.execute(
        text(
            "INSERT INTO llm_calls (stage, provider, model, input_hash, status, prompt_tokens, "
            "completion_tokens, cost_usd, latency_ms, response_json, battle_id) "
            "VALUES ('extract', 'claude_code', 'claude-code-test', 'h1', 'ok', 100, 20, 0.0012, "
            "350, "
            "CAST('{\"troops\": 1}' AS JSONB), :b)"
        ),
        {"b": ids["b0"]},
    )
    conn.execute(
        text(
            "INSERT INTO llm_calls (stage, provider, model, input_hash, status, error) "
            "VALUES ('classify', 'anthropic', 'claude-test', 'h2', 'api_error', 'timeout')"
        )
    )
    for k, (status, err) in enumerate(((200, None), (404, "not found"))):
        conn.execute(
            text(
                "INSERT INTO crawl_log (url, fetched_at, http_status, extraction_method, "
                "errors, battle_id) VALUES (:u, now() - make_interval(days => :d), :h, "
                "'infobox_parser', :e, :b)"
            ),
            {"u": f"https://example.org/{k}", "d": k, "h": status, "e": err, "b": ids["b0"]},
        )


def seed_database(engine: Engine) -> dict[str, int]:
    """Fill an empty, schema-applied database with the synthetic dataset.

    Args:
        engine: A read-write engine on a database with the schema applied.

    Returns:
        Row ids by key (``b0``..``b2`` battles, ``s00``..``s21`` sides, sources, runs).
    """
    ids: dict[str, int] = {}
    with engine.begin() as conn:
        ids.update(_seed_sources_and_runs(conn))
        _seed_battles(conn, ids)
        _seed_people_and_reports(conn, ids)
        _seed_logs(conn, ids)
    return ids


def write_data_root(root: Path, side_ids: list[int], battle_ids: list[int]) -> None:
    """Write imputed rounds, a gate report, a resolution log and raw dirs.

    Args:
        root: The data root to populate.
        side_ids: Side ids to put in the round files.
        battle_ids: Battle id of each side, same order.
    """
    rng = np.random.default_rng(0)
    for scale in ("neutral", "published"):
        scale_dir = root / "imputed" / scale
        for n in range(N_ROUNDS):
            directory = scale_dir / f"round_{n}"
            directory.mkdir(parents=True)
            lines = ["side_id,battle_id,log_troops,is_imputed,scale"]
            for k, (sid, bid) in enumerate(zip(side_ids, battle_ids, strict=True)):
                imputed = k == len(side_ids) - 1
                lines.append(
                    f"{sid},{bid},{9.0 + k * 0.2 + rng.normal(0, 0.3 if imputed else 0)},"
                    f"{imputed},{scale}"
                )
            (directory / "sides.csv").write_text("\n".join(lines), encoding="utf-8")
            (directory / "manifest.json").write_text(
                json.dumps({"run_id": 2, "scale": scale, "round": n}), encoding="utf-8"
            )
        report = {
            "run_id": 2,
            "scale": scale,
            "rounds_written": True,
            "gates": [
                {
                    "name": "completeness",
                    "passed": True,
                    "severity": "error",
                    "actual": "1.0",
                    "threshold": ">= 1",
                    "message": "",
                }
            ],
        }
        (scale_dir / "gate_report.json").write_text(json.dumps(report), encoding="utf-8")

    processed = root / "processed"
    (processed / "reconcile").mkdir(parents=True)
    np.savez(
        processed / "reconcile" / "neutral_draws_run1.npz",
        run_id=np.int64(1),
        side_id=np.asarray(side_ids, dtype=np.int64),
        battle_id=np.asarray(battle_ids, dtype=np.int64),
        draws=rng.normal(9.0, 0.3, size=(50, len(side_ids))),
        m0_mean=np.float64(9.0),
        s0_mean=np.float64(0.3),
    )
    rows = [
        {"group_key": "g1", "status": "linked", "method": "exact", "display_name": "Miltiades"},
        {"group_key": "g2", "status": "unresolved", "method": "fuzzy", "display_name": "X"},
    ]
    (processed / "resolution_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n{not json\n", encoding="utf-8"
    )
    for sub in ("battles_html", "wikidata"):
        (root / "raw" / sub).mkdir(parents=True)
        (root / "raw" / sub / "a.json").write_text("{}", encoding="utf-8")


def main() -> None:
    """Reseed ``DATABASE_URL`` and write a data root (CLI helper)."""
    root = Path(sys.argv[1])
    engine = get_engine()
    apply_schema(engine, drop_existing=True)
    ids = seed_database(engine)
    keys = [k for k in ids if k.startswith("s") and len(k) == 3 and k[1:].isdigit()]
    write_data_root(root, [ids[k] for k in keys], [ids[f"b{k[1]}"] for k in keys])
    engine.dispose()
    print(f"seeded; data root at {root}")


if __name__ == "__main__":
    main()
