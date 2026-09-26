-- ===========================================================================
-- S15 mixture recipe -- phase3: the annealing blend (60% Pool B / 40% Pool R)
-- ===========================================================================
-- Curriculum role: phase 3 re-blends SFT-grade data from both pools after the
-- RL stage. Composition: 60% Pool B / 40% Pool R, survivors only.
--
-- No difficulty_band filter (the contract's "where meaningful"): bands gate
-- the RL phase (phase2) alone. Phase 3 deliberately mixes rows of every band
-- -- SFT/annealing material includes the pass-1.0 and unlabelled rows that an
-- RL mixture must exclude.
--
-- Pool-B budget: the realized Pool-R count fixes it,
--     n_b = round(n_r * 0.60 / 0.40),
-- split by the Pool-B weights and extended exactly like phase 1: a source
-- whose target exceeds its survivor count is filled by cycling its random()
-- permutation (sampling with replacement). Integer rounding can shift the
-- realized blend by a row or two per source; the spec header records the
-- realized counts, which is what the manifest publishes.
--
-- Executed by run_mixture(run_id, sql_dir, "phase3") over the `corpus` view
-- (read_parquet([<snapshot parts>], union_by_name = true)); the recipe names
-- no files. See phase1.sql for the full RNG note (setseed vs "SET seed").

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

WITH pool_r(source) AS (VALUES
    ('medical_r1_distill'),
    ('m23k_tokenized'),
    ('medreason'),
    ('huatuo_o1_reasoning')
),
pool_b(source, weight) AS (VALUES
    ('ii_medical_reasoning_sft',   CAST(0.55 AS DOUBLE)),
    ('finemed_sft',                CAST(0.25 AS DOUBLE)),
    ('chatdoctor_healthcaremagic', CAST(0.15 AS DOUBLE)),
    ('generalthought_biology',     CAST(0.05 AS DOUBLE))
),
r_side AS (
    SELECT id, source FROM corpus
    WHERE source IN (SELECT source FROM pool_r)
      AND survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
),
r_total AS (
    SELECT count(*)::BIGINT AS n FROM r_side
),
b_budget AS (
    -- 60% B / 40% R: the realized R count fixes the B budget
    SELECT CAST(round(n * 0.60 / 0.40) AS BIGINT) AS total FROM r_total
),
b_surv AS (
    SELECT id, source FROM corpus
    WHERE source IN (SELECT source FROM pool_b)
      AND survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
),
b_counts AS (
    SELECT source, count(*)::BIGINT AS n FROM b_surv GROUP BY source
),
b_targets AS (
    -- budget split by the Pool-B weights (they sum to 1)
    SELECT w.source, CAST(round(w.weight * b.total) AS BIGINT) AS target
    FROM pool_b w CROSS JOIN b_budget b
),
b_ranked AS (
    SELECT id, source,
           row_number() OVER (PARTITION BY source ORDER BY random()) AS rn
    FROM b_surv
),
b_seq AS (
    SELECT unnest(generate_series(
        1, CAST(coalesce((SELECT max(target) FROM b_targets), 0) AS BIGINT))) AS i
),
b_draws AS (
    SELECT t.source, s.i, c.n
    FROM b_targets t
    JOIN b_counts c USING (source)
    JOIN b_seq s ON s.i <= t.target
    WHERE c.n > 0
),
b_picked AS (
    SELECT d.source, r.id
    FROM b_draws d
    JOIN b_ranked r ON r.source = d.source AND r.rn = ((d.i - 1) % d.n) + 1
)
SELECT id, source
FROM (
    SELECT id, source FROM r_side
    UNION ALL
    SELECT id, source FROM b_picked
)
ORDER BY random();
