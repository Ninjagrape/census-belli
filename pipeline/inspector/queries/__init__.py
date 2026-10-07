"""Pure ``(conn, ...) -> DataFrame`` queries, one module per concern.

Every function takes an open (read-only) connection and returns a DataFrame;
none of them write, cache, or touch Streamlit.
"""

from pipeline.inspector.queries.battle import (
    battle_casualty_reports,
    battle_commanders,
    battle_crawl_log,
    battle_llm_calls,
    battle_missing_data,
    battle_row,
    battle_sides,
    battle_troop_reports,
    battle_year_bounds,
    search_battles,
)
from pipeline.inspector.queries.classify import (
    attribution_weights,
    defender_coverage,
    imputed_rows,
    missing_data_matrix,
    review_queue,
    role_counts,
    side_weight_sums,
)
from pipeline.inspector.queries.crawl import (
    battles_by_century,
    crawl_errors,
    crawl_over_time,
    crawl_status_counts,
    llm_summary,
    null_fraction_battle_sides,
    null_fraction_battles,
    troop_report_breakdown,
)
from pipeline.inspector.queries.overview import (
    latest_model_runs,
    model_runs,
    skill_estimates,
    table_counts,
)
from pipeline.inspector.queries.reconcile import (
    casualty_estimates,
    casualty_reports_detail,
    reconcile_sides,
    source_bias,
    troop_reports_detail,
)
from pipeline.inspector.queries.resolve import (
    commanders_per_side,
    generals_missing_wikidata,
    generals_summary,
)

__all__ = [
    "attribution_weights",
    "battle_casualty_reports",
    "battle_commanders",
    "battle_crawl_log",
    "battle_llm_calls",
    "battle_missing_data",
    "battle_row",
    "battle_sides",
    "battle_troop_reports",
    "battle_year_bounds",
    "battles_by_century",
    "casualty_estimates",
    "casualty_reports_detail",
    "commanders_per_side",
    "crawl_errors",
    "crawl_over_time",
    "crawl_status_counts",
    "defender_coverage",
    "generals_missing_wikidata",
    "generals_summary",
    "imputed_rows",
    "latest_model_runs",
    "llm_summary",
    "missing_data_matrix",
    "model_runs",
    "null_fraction_battle_sides",
    "null_fraction_battles",
    "reconcile_sides",
    "review_queue",
    "role_counts",
    "search_battles",
    "side_weight_sums",
    "skill_estimates",
    "source_bias",
    "table_counts",
    "troop_report_breakdown",
    "troop_reports_detail",
]
