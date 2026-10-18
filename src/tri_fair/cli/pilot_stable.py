"""Four-dataset v8.5 screening with identical explicit controls; dry-run default."""

from tri_fair.cli.pilot import main as pilot_main

VARIANT = "Tri-Fair-v8.5"
DEFAULT_METHODS = (VARIANT, "Tri-Fair-v8.3", "NSGAII-PO-Fair")


def main(argv=None):
    return pilot_main(
        argv,
        variant_name=VARIANT,
        namespace_prefix="tri_fair_v85_pilot",
        default_methods=DEFAULT_METHODS,
        defaults=dict(
            budget=2000000,
            n_init_prompts=4,
            max_few_shot_examples=2,
            crossovers_per_iteration=1,
            final_search_candidates=4,
            final_search_reserve=0.30,
            experiment_name="v85_stable_search_pilot",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
