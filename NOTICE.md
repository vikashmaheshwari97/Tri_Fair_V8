# Provenance

This is a structural refactor of the Tri-Fair source supplied in `vikashmaheshwari97/Tri_Fair` at commit `5f748223b8c6c2f62023ded41ec898b95557df9c`, followed by the v8 upgrade prepared in this session (source release commit `f7e853e`). The core prompt-search implementation builds on the existing MO-CAPO-derived code and Promptolution. Reuse of these implementations is intentional; the clean layout is not a claim that every algorithm/metric was invented or newly authored here.

The current implementation consolidates historical optimizer behavior into semantic components and a single current optimizer. Old experiment scripts, stored split manifests, figure scripts, historical protocol bundles and historical optimizer classes are omitted. Model weights, datasets and dependencies are obtained separately under their respective terms. The research repository's owner should choose an appropriate release license before public redistribution.

Upstream references:

- https://github.com/vikashmaheshwari97/Tri_Fair
- https://github.com/mo374z/mo-capo
- https://github.com/automl/promptolution
- https://github.com/anyoptimization/pymoo

The clean package has no GEPA/DSPy baseline reproduction and makes no new measured LLM-performance claim.
