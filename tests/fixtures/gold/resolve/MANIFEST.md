# Resolve Gold Set

Agent-labelled, unreviewed. Every Q-id was verified against live Wikidata
in sessions recorded in handover.md (§13, §14, §16), not recalled from memory.

## Provenance

- Created: 2026-09-21
- Labelled by: Claude (agent), from handover.md verified live lookups
- Reviewed by: **not yet reviewed by a human**
- Candidate data: synthetic fixtures built from handover.md descriptions of
  live SPARQL results, not raw endpoint captures

## Coverage

15 mentions across 12 battles. Covers:
- Exact match (Napoleon, Caesar, Alexander, Darius, Leonidas, Scipio)
- Alias match (Duke of Wellington)
- mul-label match (Horatio Nelson)
- Date gate separation (Hannibal at Lissa 1811 vs Barca)
- MAX_PLAUSIBLE_AGE gate (Q1576150, half-dated ancient)
- Genuine ambiguity (Yi Sun-sin)
- Placeholder names (Unknown, various)
- Two spellings of one person (Agrippa / Marcus Vipsanius Agrippa)
- BC dates (Actium, Cannae, Zama, Alesia, Thermopylae, Gaugamela)

## Limitations

This is not a gold standard. It is an agent-labelled fixture set whose
candidates are synthetic rather than captured from the live endpoint. A
real gold set needs:

1. Captured SPARQL response bodies from the live endpoint
2. Human review of every expected verdict
3. Coverage of the hard cases: commanders known only by a title whose
   Wikidata label is longer than the mention, ancient commanders with
   contested identities, and names shared by more than two people alive
   in the same era
