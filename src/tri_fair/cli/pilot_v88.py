"""Tri-Fair full-coverage redesign and five matched comparison methods."""

from tri_fair.cli.pilot import main as pilot_main

VARIANT = "Tri-Fair-v8.8"
DEFAULT_METHODS = (
    VARIANT,
    "NSGAII-PO-Fair",
    "NSGAIII-PO-Fair",
    "SPEA2-PO-Fair",
    "RandomSearch-PO-Fair",
)


def main(argv=None):
    return pilot_main(
        argv,
        variant_name=VARIANT,
        namespace_prefix="tri_fair_v88_pilot",
        default_methods=DEFAULT_METHODS,
        defaults=dict(
            budget=2000000,
            n_init_prompts=4,
            max_few_shot_examples=2,
            crossovers_per_iteration=1,
            final_search_candidates=4,
            final_search_reserve=0.30,
            reclaim_audit_reserve=True,
            experiment_name="tri_fair_v88_rendered_edits_pilot",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
