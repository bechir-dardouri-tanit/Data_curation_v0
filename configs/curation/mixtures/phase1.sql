-- ===========================================================================
-- S15 mixture recipe -- phase1: the SFT starter mix (Pool B, survivors only)
-- ===========================================================================
-- Curriculum role: phase 1 of the 5-phase training curriculum. The four
-- Pool-B SFT sources, mixed 55 / 25 / 15 / 5 by weight:
--
--   ii_medical_reasoning_sft      55%   largest clean reasoning-SFT pool
--   finemed_sft                   25%
--   chatdoctor_healthcaremagic    15%
--   generalthought_biology         5%
--
-- Executed by medrl.curation.stages.mixture.run_mixture(run_id, sql_dir,
-- "phase1"), which first creates the view this file queries:
--
--   CREATE VIEW corpus AS SELECT * FROM read_parquet([<snapshot parts>],
--                                                    union_by_name = true);
--
-- The recipe names no files: it is executable against any run's snapshot.
--
-- ANCHOR RULE (documented per the mixture contract): the mixture's absolute
-- scale is anchored on the SMALLEST surviving Pool-B source (ties broken by
-- source name, so the rule is deterministic). The anchor contributes all of
-- its survivors; every other source's target is
--
--     target_i = round(w_i * n_anchor / w_anchor)
--
-- i.e. the Pool-B weights expressed against the anchor's full survivor count.
-- A target above a source's survivor count is met by cycling that source's
-- random() permutation (sampling WITH replacement -- instance i takes rank
-- ((i-1) % n) + 1); a target below it takes a uniform random subset (a prefix
-- of the same permutation). Rationale: never invent rows, never dilute the
-- scarcest source, and let the abundant sources carry their weight. A source
-- with zero survivors contributes zero rows, however large its weight, and
-- the realized counts land in the spec header for honest reporting.

-- RNG note: DuckDB's random() draws from a session-seeded generator; every
-- recipe pins the session seed so an assembly is repeatable for a given
-- duckdb build + thread count (not a cross-version guarantee -- the manifest
-- records realized counts). The mixture contract's "SET seed=42" is spelled
--   SELECT setseed(0.42);
-- because duckdb 1.5.5 has no `seed` configuration parameter: setseed() is
-- this engine's session-seed mechanism, and 42 lands in its [-1, 1] domain
-- as 0.42.
SELECT setseed(0.42);

-- The survivor predicate -- a row with NO quality, dedup, contamination,
-- answer or grounding flag -- spelled exactly once as a SQL macro (the
-- mixture contract's "comment macro", made executable so the file runs as
-- shipped; duckdb macros take the flags as arguments because parameterless
-- macros cannot reference columns). NULL flags fail closed: they exclude.
CREATE OR REPLACE MACRO survivor(
    empty, length, truncated, repetition, lang, encoding, refusal,
    dup_exact, dup_minhash, dup_semantic, dup_concept,
    contam_ngram, contam_semantic, contam_concept,
    answer_wrong, answer_right_reasoning_contradicts, kg_contradicted
) AS (
    NOT empty AND NOT length AND NOT truncated AND NOT repetition AND NOT lang
    AND NOT encoding AND NOT refusal
    AND NOT dup_exact AND NOT dup_minhash AND NOT dup_semantic AND NOT dup_concept
    AND NOT contam_ngram AND NOT contam_semantic AND NOT contam_concept
    AND NOT answer_wrong AND NOT answer_right_reasoning_contradicts
    AND NOT kg_contradicted
);

WITH pool_b(source, weight) AS (VALUES
    ('ii_medical_reasoning_sft',   CAST(0.55 AS DOUBLE)),
    ('finemed_sft',                CAST(0.25 AS DOUBLE)),
    ('chatdoctor_healthcaremagic', CAST(0.15 AS DOUBLE)),
    ('generalthought_biology',     CAST(0.05 AS DOUBLE))
),
surv AS (
    SELECT id, source FROM corpus
    WHERE source IN (SELECT source FROM pool_b)
      AND survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
),
counts AS (
    SELECT source, count(*)::BIGINT AS n FROM surv GROUP BY source
),
anchor AS (
    -- smallest surviving source; source name breaks ties
    SELECT c.n AS n_anchor, w.weight AS w_anchor
    FROM counts c JOIN pool_b w USING (source)
    ORDER BY c.n ASC, c.source ASC
    LIMIT 1
),
targets AS (
    SELECT w.source,
           CAST(round(w.weight * a.n_anchor / a.w_anchor) AS BIGINT) AS target
    FROM pool_b w CROSS JOIN anchor a
),
ranked AS (
    SELECT id, source,
           row_number() OVER (PARTITION BY source ORDER BY random()) AS rn
    FROM surv
),
seq AS (
    SELECT unnest(generate_series(
        1, CAST(coalesce((SELECT max(target) FROM targets), 0) AS BIGINT))) AS i
),
draws AS (
    -- a source with zero survivors joins to nothing here and contributes
    -- nothing, whatever its weight implies
    SELECT t.source, s.i, c.n
    FROM targets t
    JOIN counts c USING (source)
    JOIN seq s ON s.i <= t.target
    WHERE c.n > 0
),
picked AS (
    SELECT d.source, r.id, d.i AS instance
    FROM draws d
    JOIN ranked r ON r.source = d.source AND r.rn = ((d.i - 1) % d.n) + 1
)
SELECT id, source, instance
FROM picked
ORDER BY random();
