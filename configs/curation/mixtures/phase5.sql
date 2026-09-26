-- ===========================================================================
-- S15 mixture recipe -- phase5: the polish set (150k II-Medical + 50k Pool R)
-- ===========================================================================
-- Curriculum role: phase 5 is the final continued-SFT polish pass over the two
-- highest-quality pools: 150k random ii_medical_reasoning_sft survivors plus
-- 50k Pool R survivors (medical_r1_distill, m23k_tokenized, medreason,
-- huatuo_o1_reasoning), roughly 3:1.
--
-- Absolute counts by design: this phase sizes the run, it does not scale with
-- the snapshot -- a smaller snapshot contributes all it has and LIMIT only
-- caps. No difficulty_band filter (see phase3: bands gate the RL phase alone).
--
-- RNG NOTE (the contract's documented one): DuckDB's random() draws from a
-- SESSION-SEEDED generator. This recipe sets the session seed once, up top.
-- The contract spells that "SET seed=42"; duckdb 1.5.5 has no `seed`
-- configuration parameter, so the same mechanism is invoked in its supported
-- spelling, and 42 lands in setseed's [-1, 1] domain:
--
--     SELECT setseed(0.42);
--
-- The seed makes the exact 150k/50k draw repeatable for a given duckdb build
-- + thread count; it is not a cross-version guarantee, and the assembly's
-- realized counts are recorded in the manifest regardless.
--
-- Executed by run_mixture(run_id, sql_dir, "phase5") over the `corpus` view
-- (read_parquet([<snapshot parts>], union_by_name = true)); the recipe names
-- no files.

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
ii AS (
    SELECT id, source FROM corpus
    WHERE source = 'ii_medical_reasoning_sft'
      AND survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
    ORDER BY random()
    LIMIT 150000
),
r AS (
    SELECT id, source FROM corpus
    WHERE source IN (SELECT source FROM pool_r)
      AND survivor(
            flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
            flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
            flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
            flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
            flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
    ORDER BY random()
    LIMIT 50000
)
SELECT id, source
FROM (
    SELECT id, source FROM ii
    UNION ALL
    SELECT id, source FROM r
)
ORDER BY random();
