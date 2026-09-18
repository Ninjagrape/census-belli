-- =============================================================================
-- MILITARY GENERAL RANKING PROJECT — DATA SCHEMA
-- =============================================================================
-- Designed to support multi-source data collection, command hierarchy
-- attribution, confidence-scored field values, and Bayesian modelling.
-- Target DB: PostgreSQL 15+ (uses JSONB, enums, generated columns)
-- =============================================================================

-- ─── ENUMS ──────────────────────────────────────────────────────────────────

CREATE TYPE battle_type AS ENUM (
    'field',
    'siege_offensive',
    'siege_defensive',
    'naval',
    'aerial',
    'combined',
    'amphibious',
    'guerrilla',
    'unknown'
);

CREATE TYPE outcome_level AS ENUM (
    'decisive_victory',
    'victory',
    'pyrrhic_victory',
    'indecisive',
    'defeat',
    'decisive_defeat'
);

CREATE TYPE command_role AS ENUM (
    'sovereign',            -- head of state present but not commanding tactically
    'supreme_commander',    -- overall strategic authority (e.g. Eisenhower on D-Day)
    'theatre_commander',    -- commands the campaign/theatre
    'field_commander',      -- commands the army/fleet in the engagement
    'subordinate',          -- commands a wing, division, or corps under another
    'nominal',              -- listed in infobox but functionally absent or political
    'unknown'
);

CREATE TYPE troop_branch AS ENUM (
    'total',
    'infantry',
    'cavalry',
    'artillery',
    'naval',
    'air',
    'armour',
    'irregular',
    'other'
);

CREATE TYPE source_type AS ENUM (
    'peer_reviewed',
    'academic_book',
    'encyclopedia',
    'primary_source',
    'wikipedia_infobox',
    'wikipedia_body',
    'wikidata',
    'dbpedia',
    'web_secondary',
    'manual_entry'
);

CREATE TYPE missingness_class AS ENUM (
    'observed',
    'mcar',      -- missing completely at random
    'mar',       -- missing at random (conditional on observables)
    'mnar',      -- missing not at random
    'unclassified'
);

CREATE TYPE extraction_method AS ENUM (
    'infobox_parser',
    'llm_extraction',
    'wikidata_sparql',
    'dbpedia_rdf',
    'manual',
    'ocr',
    'structured_db'    -- e.g. Correlates of War dataset
);


-- ─── SOURCES & PROVENANCE ───────────────────────────────────────────────────

CREATE TABLE sources (
    source_id       SERIAL PRIMARY KEY,
    source_type     source_type NOT NULL,
    title           TEXT,
    author          TEXT,
    year            INT,
    url             TEXT,
    doi             TEXT,
    isbn            TEXT,
    -- how reliable is this source category for troop numbers specifically
    -- (learned/calibrated over time, not fixed)
    troop_bias_mu   DOUBLE PRECISION DEFAULT 0.0,   -- estimated systematic bias (log scale)
    troop_bias_sd   DOUBLE PRECISION DEFAULT 1.0,   -- uncertainty on that bias
    credibility     DOUBLE PRECISION DEFAULT 0.5     -- 0..1, prior on source quality
        CHECK (credibility BETWEEN 0.0 AND 1.0),
    raw_text_cache  TEXT,                            -- cached extraction from crawl
    fetched_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_sources_type ON sources (source_type);
CREATE INDEX idx_sources_url  ON sources (url) WHERE url IS NOT NULL;


-- ─── GENERALS (entity-resolved) ─────────────────────────────────────────────

CREATE TABLE generals (
    general_id      SERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    wikidata_id     TEXT UNIQUE,           -- e.g. 'Q517' for Napoleon
    wikipedia_url   TEXT,
    born            DATE,
    died            DATE,
    nationality     TEXT,                  -- primary, simplified
    era             TEXT,                  -- e.g. 'Napoleonic', 'Ancient Rome'
    -- biographical metadata useful as model features
    highest_rank    TEXT,
    years_active    INT4RANGE,             -- range of active years
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE general_aliases (
    alias_id        SERIAL PRIMARY KEY,
    general_id      INT NOT NULL REFERENCES generals(general_id),
    alias_name      TEXT NOT NULL,
    language        TEXT DEFAULT 'en',     -- alias might be in a different language
    source_id       INT REFERENCES sources(source_id),
    is_primary      BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_aliases_name    ON general_aliases (alias_name);
CREATE INDEX idx_aliases_general ON general_aliases (general_id);


-- ─── WARS & CAMPAIGNS ───────────────────────────────────────────────────────

CREATE TABLE wars (
    war_id          SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    wikidata_id     TEXT UNIQUE,
    start_year      INT,
    end_year        INT,
    region          TEXT,
    notes           TEXT
);

CREATE TABLE campaigns (
    campaign_id     SERIAL PRIMARY KEY,
    war_id          INT REFERENCES wars(war_id),
    name            TEXT NOT NULL,
    wikidata_id     TEXT UNIQUE,
    start_date      DATE,
    end_date        DATE,
    notes           TEXT
);


-- ─── BATTLES ────────────────────────────────────────────────────────────────

CREATE TABLE battles (
    battle_id       SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    wikidata_id     TEXT UNIQUE,
    wikipedia_url   TEXT,
    campaign_id     INT REFERENCES campaigns(campaign_id),
    war_id          INT REFERENCES wars(war_id),

    -- temporal
    date_start      DATE,
    date_end        DATE,
    date_precision  TEXT DEFAULT 'day',    -- 'day', 'month', 'year', 'decade', 'century'

    -- spatial
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    location_name   TEXT,

    -- classification
    battle_type     battle_type DEFAULT 'unknown',
    terrain         TEXT[],                -- e.g. {'mountainous', 'forested'}
    fortified       BOOLEAN,              -- was a fortification involved
    weather         TEXT,                  -- extracted from article if available

    -- data quality
    data_quality_score  DOUBLE PRECISION DEFAULT 0.5
        CHECK (data_quality_score BETWEEN 0.0 AND 1.0),
    n_sources           INT DEFAULT 0,    -- how many independent sources cover this battle
    needs_review        BOOLEAN DEFAULT FALSE,
    review_notes        TEXT,

    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_battles_war      ON battles (war_id);
CREATE INDEX idx_battles_campaign ON battles (campaign_id);
CREATE INDEX idx_battles_date     ON battles (date_start);
CREATE INDEX idx_battles_quality  ON battles (data_quality_score);


-- ─── BATTLE SIDES ───────────────────────────────────────────────────────────
-- Each battle has 2+ sides. This replaces the original flat "combatant1/combatant2"
-- and allows multi-faction battles.

CREATE TABLE battle_sides (
    side_id         SERIAL PRIMARY KEY,
    battle_id       INT NOT NULL REFERENCES battles(battle_id),
    side_label      TEXT NOT NULL,         -- e.g. 'French Empire', 'Coalition Forces'
    polity          TEXT,                  -- standardised polity name
    wikidata_id     TEXT,                  -- wikidata entity for the polity

    -- outcome for this side
    outcome         outcome_level,
    outcome_source  INT REFERENCES sources(source_id),

    -- best-estimate troop totals (derived from troop_reports, not entered directly)
    est_troops_total     DOUBLE PRECISION,
    est_troops_total_lo  DOUBLE PRECISION,  -- 95% CI lower bound
    est_troops_total_hi  DOUBLE PRECISION,  -- 95% CI upper bound
    est_casualties       DOUBLE PRECISION,
    est_casualties_lo    DOUBLE PRECISION,
    est_casualties_hi    DOUBLE PRECISION,

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_sides_battle ON battle_sides (battle_id);


-- ─── BATTLE COMMANDERS ──────────────────────────────────────────────────────
-- The critical junction table. Each row links one general to one side of one
-- battle, with their command role and position in the hierarchy.

CREATE TABLE battle_commanders (
    bc_id               SERIAL PRIMARY KEY,
    battle_id           INT NOT NULL REFERENCES battles(battle_id),
    side_id             INT NOT NULL REFERENCES battle_sides(side_id),
    general_id          INT NOT NULL REFERENCES generals(general_id),

    -- command role and hierarchy
    command_role        command_role NOT NULL DEFAULT 'unknown',
    reports_to_bc_id    INT REFERENCES battle_commanders(bc_id),  -- parent in command chain
    hierarchy_rank      INT DEFAULT 0,     -- 0 = top commander on this side, 1 = direct report, etc.

    -- attribution (may be learned by the model or set by extraction)
    attribution_weight  DOUBLE PRECISION DEFAULT 1.0  -- 0..1, how much credit this person gets
        CHECK (attribution_weight BETWEEN 0.0 AND 1.0),
    attribution_method  TEXT DEFAULT 'equal',          -- 'equal', 'role_heuristic', 'model_inferred', 'manual'

    -- provenance: how do we know this person commanded here
    source_id           INT REFERENCES sources(source_id),
    extraction_method   extraction_method,
    confidence          DOUBLE PRECISION DEFAULT 0.5
        CHECK (confidence BETWEEN 0.0 AND 1.0),

    -- extracted context from article text explaining their role
    role_evidence       TEXT,              -- e.g. "Agrippa commanded the fleet while Octavian remained on shore"

    created_at          TIMESTAMPTZ DEFAULT now(),

    UNIQUE (battle_id, side_id, general_id)
);

CREATE INDEX idx_bc_general ON battle_commanders (general_id);
CREATE INDEX idx_bc_battle  ON battle_commanders (battle_id);
CREATE INDEX idx_bc_side    ON battle_commanders (side_id);
CREATE INDEX idx_bc_role    ON battle_commanders (command_role);


-- ─── TROOP REPORTS ──────────────────────────────────────────────────────────
-- Multiple sources can report different troop numbers for the same side.
-- The source-disagreement model (Phase 2d) consumes these to produce the
-- best estimates stored on battle_sides.

CREATE TABLE troop_reports (
    report_id       SERIAL PRIMARY KEY,
    side_id         INT NOT NULL REFERENCES battle_sides(side_id),
    source_id       INT NOT NULL REFERENCES sources(source_id),

    branch          troop_branch NOT NULL DEFAULT 'total',
    reported_value  DOUBLE PRECISION NOT NULL,

    -- what does this number actually represent
    scope           TEXT DEFAULT 'engaged',  -- 'engaged', 'available', 'theatre_strength', 'on_paper', 'unknown'
    is_estimate     BOOLEAN DEFAULT FALSE,   -- source itself says "approximately"
    is_upper_bound  BOOLEAN DEFAULT FALSE,   -- source says "up to X"
    is_lower_bound  BOOLEAN DEFAULT FALSE,   -- source says "at least X"

    -- extraction metadata
    extraction_method   extraction_method,
    extracted_context   TEXT,               -- the sentence/paragraph the number came from
    page_or_section     TEXT,               -- where in the source

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_troop_side   ON troop_reports (side_id);
CREATE INDEX idx_troop_source ON troop_reports (source_id);
CREATE INDEX idx_troop_branch ON troop_reports (branch);


-- ─── CASUALTY REPORTS ───────────────────────────────────────────────────────

CREATE TABLE casualty_reports (
    report_id       SERIAL PRIMARY KEY,
    side_id         INT NOT NULL REFERENCES battle_sides(side_id),
    source_id       INT NOT NULL REFERENCES sources(source_id),

    casualty_type   TEXT NOT NULL DEFAULT 'total',  -- 'killed', 'wounded', 'captured', 'missing', 'total'
    reported_value  DOUBLE PRECISION NOT NULL,
    is_estimate     BOOLEAN DEFAULT FALSE,

    extraction_method   extraction_method,
    extracted_context   TEXT,

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_casualty_side ON casualty_reports (side_id);


-- ─── MISSING DATA TRACKING ─────────────────────────────────────────────────
-- Tracks which fields are missing for which battles, their missingness class,
-- and what imputation was applied.

CREATE TABLE missing_data_log (
    log_id              SERIAL PRIMARY KEY,
    battle_id           INT NOT NULL REFERENCES battles(battle_id),
    side_id             INT REFERENCES battle_sides(side_id),
    field_name          TEXT NOT NULL,          -- e.g. 'troop_total', 'outcome', 'date'
    missingness_class   missingness_class DEFAULT 'unclassified',

    -- imputation tracking
    was_imputed         BOOLEAN DEFAULT FALSE,
    imputation_method   TEXT,                   -- e.g. 'mice_pmm', 'era_polity_prior', 'manual'
    imputed_value       DOUBLE PRECISION,
    imputed_lo          DOUBLE PRECISION,       -- 95% CI
    imputed_hi          DOUBLE PRECISION,
    imputation_round    INT,                    -- which imputation dataset (for multiple imputation)

    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_missing_battle ON missing_data_log (battle_id);
CREATE INDEX idx_missing_field  ON missing_data_log (field_name);


-- ─── CRAWL & EXTRACTION METADATA ───────────────────────────────────────────

CREATE TABLE crawl_log (
    crawl_id        SERIAL PRIMARY KEY,
    url             TEXT NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL,
    http_status     INT,
    content_hash    TEXT,                  -- detect changes on re-crawl
    extraction_method  extraction_method,
    fields_extracted   INT DEFAULT 0,
    errors          TEXT,
    battle_id       INT REFERENCES battles(battle_id),  -- which battle this crawl targeted
    source_id       INT REFERENCES sources(source_id)
);

CREATE INDEX idx_crawl_url    ON crawl_log (url);
CREATE INDEX idx_crawl_battle ON crawl_log (battle_id);


-- ─── MODEL OUTPUTS ──────────────────────────────────────────────────────────
-- Stores results from different model runs so you can compare specifications.

CREATE TABLE model_runs (
    run_id          SERIAL PRIMARY KEY,
    run_name        TEXT NOT NULL,
    model_type      TEXT NOT NULL,         -- e.g. 'bradley_terry_hierarchical', 'linear_baseline'
    config          JSONB NOT NULL,        -- full model config (priors, covariates, imputation method)
    imputation_set  INT,                   -- which multiple-imputation dataset, NULL = observed only
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    diagnostics     JSONB,                 -- rhat, ess, divergences, etc.
    notes           TEXT
);

CREATE TABLE general_skill_estimates (
    estimate_id     SERIAL PRIMARY KEY,
    run_id          INT NOT NULL REFERENCES model_runs(run_id),
    general_id      INT NOT NULL REFERENCES generals(general_id),

    -- posterior summary for latent skill parameter
    skill_mean      DOUBLE PRECISION NOT NULL,
    skill_median    DOUBLE PRECISION,
    skill_sd        DOUBLE PRECISION NOT NULL,
    skill_ci_lo     DOUBLE PRECISION NOT NULL,  -- 95% credible interval
    skill_ci_hi     DOUBLE PRECISION NOT NULL,
    skill_ci90_lo   DOUBLE PRECISION,           -- 90% CI for tighter comparison
    skill_ci90_hi   DOUBLE PRECISION,

    -- WAR aggregates
    total_war       DOUBLE PRECISION,
    war_per_battle  DOUBLE PRECISION,
    war_sd          DOUBLE PRECISION,           -- uncertainty on total WAR

    n_battles       INT NOT NULL,
    rank            INT,                        -- within this model run

    UNIQUE (run_id, general_id)
);

CREATE INDEX idx_skill_run     ON general_skill_estimates (run_id);
CREATE INDEX idx_skill_general ON general_skill_estimates (general_id);
CREATE INDEX idx_skill_rank    ON general_skill_estimates (run_id, rank);

CREATE TABLE battle_war_details (
    detail_id       SERIAL PRIMARY KEY,
    run_id          INT NOT NULL REFERENCES model_runs(run_id),
    battle_id       INT NOT NULL REFERENCES battles(battle_id),
    general_id      INT NOT NULL REFERENCES generals(general_id),

    -- per-battle WAR breakdown
    war_contribution    DOUBLE PRECISION NOT NULL,
    win_prob_model      DOUBLE PRECISION,   -- P(win) predicted by the model
    win_prob_baseline   DOUBLE PRECISION,   -- P(win) for a replacement-level general
    actual_outcome      DOUBLE PRECISION,   -- 1, 0.5 (draw), 0
    attribution_weight  DOUBLE PRECISION,   -- how much of this side's outcome is attributed to this general

    UNIQUE (run_id, battle_id, general_id)
);

CREATE INDEX idx_bwd_run     ON battle_war_details (run_id);
CREATE INDEX idx_bwd_general ON battle_war_details (general_id);


-- ─── VIEWS ──────────────────────────────────────────────────────────────────

-- Quick lookup: for any battle, show all commanders with roles and troop context
CREATE VIEW v_battle_overview AS
SELECT
    b.battle_id,
    b.name AS battle_name,
    b.date_start,
    b.battle_type,
    bs.side_label,
    bs.outcome,
    bs.est_troops_total,
    g.canonical_name AS general_name,
    bc.command_role,
    bc.hierarchy_rank,
    bc.attribution_weight,
    bc.role_evidence
FROM battles b
JOIN battle_sides bs   ON bs.battle_id = b.battle_id
JOIN battle_commanders bc ON bc.side_id = bs.side_id
JOIN generals g        ON g.general_id = bc.general_id
ORDER BY b.date_start, b.battle_id, bs.side_id, bc.hierarchy_rank;

-- Source agreement: for each side's troop total, how many sources and how much spread
CREATE VIEW v_troop_source_agreement AS
SELECT
    tr.side_id,
    b.name AS battle_name,
    bs.side_label,
    tr.branch,
    COUNT(*)                             AS n_reports,
    AVG(tr.reported_value)               AS mean_reported,
    STDDEV(tr.reported_value)            AS sd_reported,
    MIN(tr.reported_value)               AS min_reported,
    MAX(tr.reported_value)               AS max_reported,
    -- coefficient of variation as a quick disagreement flag
    CASE WHEN AVG(tr.reported_value) > 0
         THEN STDDEV(tr.reported_value) / AVG(tr.reported_value)
         ELSE NULL END                   AS cv
FROM troop_reports tr
JOIN battle_sides bs ON bs.side_id = tr.side_id
JOIN battles b       ON b.battle_id = bs.battle_id
GROUP BY tr.side_id, b.name, bs.side_label, tr.branch;

-- Generals with low-confidence command attributions (review queue)
CREATE VIEW v_attribution_review_queue AS
SELECT
    g.canonical_name,
    b.name AS battle_name,
    bc.command_role,
    bc.confidence,
    bc.role_evidence,
    bc.attribution_method
FROM battle_commanders bc
JOIN generals g ON g.general_id = bc.general_id
JOIN battles b  ON b.battle_id  = bc.battle_id
WHERE bc.confidence < 0.5
   OR bc.command_role = 'unknown'
ORDER BY bc.confidence ASC;
