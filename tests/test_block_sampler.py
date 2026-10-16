import pandas as pd

from tri_fair.fairness.block_sampler import MANIFEST_ID, build_fairness_blocks


def test_bbq_blocks_have_one_quartet_per_category():
    rows = []
    categories = [f"cat{i}" for i in range(11)]
    for block in range(2):
        for category in categories:
            template = f"{category}:{block}"
            for variant in range(4):
                rows.append(
                    {
                        MANIFEST_ID: f"{template}:{variant}",
                        "category": category,
                        "template_id": template,
                    }
                )
    frame = pd.DataFrame(rows)
    blocks = build_fairness_blocks(
        frame,
        dataset_alias="bbq",
        block_size=44,
        seed=42,
    )
    assert len(blocks) == 2
    assert all(len(block) == 44 for block in blocks)
    assert len({item for block in blocks for item in block}) == len(frame)


def test_bios_blocks_cover_every_profession_gender_cell():
    rows = []
    for repetition in range(4):
        for profession in range(28):
            for gender in (0, 1):
                rows.append(
                    {
                        MANIFEST_ID: f"{repetition}:{profession}:{gender}",
                        "profession_id": profession,
                        "gender": gender,
                    }
                )
    frame = pd.DataFrame(rows)
    blocks = build_fairness_blocks(
        frame,
        dataset_alias="bias_in_bios",
        block_size=112,
        seed=42,
    )
    assert len(blocks) == 2
    assert all(len(block) == 112 for block in blocks)
