"""Measured v9.1 plots; no transformation of experimental results."""
import plot_v89 as base
base.STYLE = {
    **base.STYLE,
    "Tri-Fair-v9.1": ("Tri-Fair v9.1 intensification", "#000000"),
    "Tri-Fair-v9.1-FullEval": ("v9.1 full evaluation control", "#0072B2"),
}
if __name__ == "__main__":
    base.main()
