# Clean package validation

72 CPU tests passed with no failures, errors or skips under Python 3.12.14 on Linux. The installable project was installed using its provided requirements file, and `pip check` passed. The machine-readable test report is `cpu_tests.xml`.

22 seeded before/after cases (11 algorithm/options × 2 seeds × 3 optimizer steps) matched complete prompts, archive membership, scores and downstream/meta token counts exactly with deterministic synthetic models. `behavior_parity.json` records each case. This validates the restructuring on those fixtures, not universal behavioral equivalence for every response, interruption or backend. The historical corrected v7.1 method is not included in the clean current-method registry.

All 60 package modules imported on CPU without the original repository. Source/tests passed Ruff and compilation. Seven CLI help paths, the CPU doctor, the 108-task plan, shell syntax and dry-run submissions were checked. PDF/SVG/PNG exporters were exercised using synthetic fixtures. GPU runtime/model smoke entry points are covered with fake predictors but were not executed against real GPUs here.

The previous Windows terminal attachment demonstrated 69 passing tests for the older bundle under Python 3.12. The new clean-package Windows installation, native PowerShell helper and GitHub-hosted Python 3.11/3.12 CI were not executed here. Run the supplied local checks before pushing, then the provided backend/optimizer smoke jobs on Rocket. No real-model performance improvement has been established.
