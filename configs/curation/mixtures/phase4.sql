-- ===========================================================================
-- S15 mixture recipe -- phase4: the preference mix (Pool P pairs)
-- ===========================================================================
-- Curriculum role: phase 4 is the DPO stage's data. Sources: finemed_dpo,
-- ii_medical_rl and chatdoctor_rl (the Pool-P preference/pair sources), taken
-- whole -- survivors only, never subsampled, because preference pairs are the
-- scarcest resource in the corpus.
--
-- meta.pair is the source-agnostic marker: either the row carries its own
-- chosen/rejected payloads under that key, or it participates in a pair group
-- identified by it. Either way its presence marks preference data, so such
-- rows are mixed even outside the three P sources. The marker is read with
-- try_cast so a foreign snapshot with malformed meta JSON fails soft (NULL ->
-- excluded) instead of aborting the assembly.
--
-- No difficulty_band filter: DPO pairs are routed by their pair structure,
-- not by their labelled pass rate.
--
-- Executed by run_mixture(run_id, sql_dir, "phase4") over the `corpus` view
-- (read_parquet([<snapshot parts>], union_by_name = true)); the recipe names
-- no files. See phase1.sql for the full RNG note (setseed vs "SET seed");
-- the seed here only shuffles assembly order, the selection is a filter.

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

SELECT id, source
FROM corpus
WHERE survivor(
        flags_empty, flags_length, flags_truncated, flags_repetition, flags_lang,
        flags_encoding, flags_refusal, flags_dup_exact, flags_dup_minhash,
        flags_dup_semantic, flags_dup_concept, flags_contam_ngram,
        flags_contam_semantic, flags_contam_concept, flags_answer_wrong,
        flags_answer_right_reasoning_contradicts, flags_kg_contradicted)
  AND (
        source IN ('finemed_dpo', 'ii_medical_rl', 'chatdoctor_rl')
        OR json_extract_string(TRY_CAST(meta AS JSON), '$.pair') IS NOT NULL
      )
ORDER BY random();
