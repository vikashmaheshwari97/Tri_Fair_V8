"""Evidence-racing v9.4 five-method pilot with shared budget admission."""

from tri_fair.cli.pilot import main as pilot_main

DEFAULT_METHODS = (
    "Tri-Fair-v9.4",
    "NSGAII-PO-Fair-Plain",
    "NSGAIII-PO-Fair-Plain",
    "SPEA2-PO-Fair-Plain",
    "RandomSearch-PO-Fair-Plain",
)


def main(argv=None):
    return pilot_main(
        argv,
        variant_name="Tri-Fair-v9.4",
        namespace_prefix="tri_fair_v94_pilot",
        default_methods=DEFAULT_METHODS,
        defaults=dict(
            budget=2000000,
            n_init_prompts=4,
            max_few_shot_examples=2,
            crossovers_per_iteration=1,
            final_search_candidates=4,
            final_search_reserve=0.20,
            reclaim_audit_reserve=False,
            budget_fill=True,
            experiment_name="tri_fair_v94_controlled_pilot",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
