"""Use established measured plots with explicit v9.0/ablation labels."""

import plot_v89 as base

base.STYLE = {
    "Tri-Fair-v9.0": ("Tri-Fair v9.0", "#000000"),
    "Tri-Fair-v9.0-NoDisparity": ("v9.0 without disparity evidence", "#0072B2"),
    "Tri-Fair-v9.0-NoCoverage": ("v9.0 without coverage survival", "#D55E00"),
    **base.STYLE,
}
if __name__ == "__main__":
    base.main()
