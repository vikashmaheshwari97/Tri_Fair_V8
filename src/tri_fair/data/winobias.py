"""Pinned official WinoBias data, converted to two-choice coreference.

All bracket annotations are removed before inference. Adjacent source sentences,
pro/anti variants and both task types share a split unit. No test labels enter
search. This is an accuracy-gap adaptation, not the original CoNLL F1 scorer.
"""

from __future__ import annotations
import hashlib
from importlib.resources import files
import json
import re
import numpy as np
import pandas as pd

REVISION = "0bce984dd081bbc10b0622f326727a024c607895"
PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}


def _payload():
    value = json.loads(
        files("tri_fair")
        .joinpath("assets/winobias_official.json")
        .read_text(encoding="utf-8")
    )
    if value["revision"] != REVISION:
        raise ValueError("WinoBias source revision mismatch")
    for name, text in value["sources"].items():
        if hashlib.sha256(text.encode()).hexdigest() != value["sha256"][name]:
            raise ValueError(f"WinoBias source checksum mismatch: {name}")
    return value


def load_winobias_sources(config=None):
    payload = _payload()
    if config is not None and config.revision != REVISION:
        raise ValueError("WinoBias configuration revision mismatch")
    vocabulary = set()
    for gender in ["male", "female"]:
        vocabulary.update(
            payload["sources"][f"{gender}_occupations.txt"].casefold().splitlines()
        )
    occupation = re.compile(
        r"\b(?:"
        + "|".join(re.escape(x) for x in sorted(vocabulary, key=lambda x: (-len(x), x)))
        + r")\b",
        re.I,
    )
    rows = []
    for name, content in payload["sources"].items():
        match = re.fullmatch(
            r"(pro|anti)_stereotyped_type([12])\.txt\.(dev|test)", name
        )
        if not match:
            continue
        condition, task_type, split = match.groups()
        for line in content.splitlines():
            number, annotated = line.split(" ", 1)
            marks = [
                " ".join(s.casefold().split())
                for s in re.findall(r"\[([^]]+)\]", annotated)
            ]
            pronouns = [s for s in marks if s in PRONOUNS]
            antecedents = [s for s in marks if s not in PRONOUNS]
            if not pronouns or len(antecedents) != 1:
                raise ValueError(f"Malformed WinoBias annotations: {name}:{number}")
            correct = occupation.search(antecedents[0])
            sentence = " ".join(annotated.replace("[", "").replace("]", "").split())
            # The two candidates are the first two occupation mentions; later
            # third parties (e.g. reporting to a manager) are distractors.
            options = list(
                dict.fromkeys(
                    m.group().casefold() for m in occupation.finditer(sentence)
                )
            )[:2]
            if (
                correct is None
                or len(options) != 2
                or correct.group().casefold() not in options
            ):
                raise ValueError(
                    f"Cannot resolve exactly two occupations: {name}:{number}"
                )
            # Stable option permutation, identical for a matched gender pair.
            pair_id = f"{split}/type{task_type}/{int(number)}"
            if hashlib.sha256(pair_id.encode()).digest()[0] % 2:
                options.reverse()
            target = "ab"[options.index(correct.group().casefold())]
            neutral = re.sub(
                r"\b(?:he|she|him|her|his|hers|himself|herself)\b",
                "<person>",
                sentence.casefold(),
            )
            rows.append(
                dict(
                    _manifest_id=f"{pair_id}/{condition}",
                    _source_split=split,
                    source_number=int(number),
                    task_type=f"type{task_type}",
                    stereotype=condition,
                    gender="female"
                    if pronouns[0] in {"she", "her", "hers", "herself"}
                    else "male",
                    pair_id=pair_id,
                    template_id=f"{split}/family{(int(number) - 1) // 2}",
                    canonical_template=neutral,
                    target=target,
                    input=f'{sentence}\nThe marked pronoun is "{pronouns[0]}". Which occupation does it refer to?\na: {options[0]}\nb: {options[1]}',
                )
            )
    frame = pd.DataFrame(rows)
    # Actual canonical duplicates, including across official splits, share one
    # connected component. Conflicting official-test components are removed
    # from development, rather than leaking through a different sentence ID.
    parent = {key: key for key in frame.template_id.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, group in frame.groupby("canonical_template", sort=True):
        keys = group.template_id.unique()
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    frame["template_id"] = frame.template_id.map(find)
    test_components = set(frame.loc[frame._source_split == "test", "template_id"])
    frame = frame[
        ~((frame._source_split == "dev") & frame.template_id.isin(test_components))
    ].copy()
    return {s: g.reset_index(drop=True) for s, g in frame.groupby("_source_split")}


def sample_units(frame, *, size, seed):
    """Sample complete connected template families, balanced across types."""
    if size % 8:
        raise ValueError("WinoBias partition sizes must be divisible by eight")
    units = list(frame.groupby("template_id", sort=True))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    # Most families contain eight rows. Larger connected duplicate families are
    # retained only when they fit whole; never split or truncate a family.
    selected = []
    for _, group in units:
        if len(group) <= size - len(selected):
            selected.extend(group._manifest_id.astype(str))
        if len(selected) == size:
            break
    if len(selected) != size:
        raise ValueError(f"Cannot sample {size} complete WinoBias family rows")
    return selected


def build_blocks(frame, *, block_size, seed):
    if block_size % 8 or len(frame) % block_size:
        raise ValueError("WinoBias blocks require whole eight-row families")
    remaining, blocks = frame.copy(), []
    for index in range(len(frame) // block_size):
        ids = sample_units(remaining, size=block_size, seed=seed + index * 1009)
        blocks.append(ids)
        remaining = remaining[~remaining._manifest_id.isin(ids)]
    return blocks


def validate_partitions(sources, partitions, blocks):
    templates = {}
    for role, split in [("dev", "dev"), ("few_shot", "dev"), ("test", "test")]:
        selected = sources[split][sources[split]._manifest_id.isin(partitions[role])]
        templates[role] = set(selected.template_id)
        # Every included family must be complete in this partition.
        complete = sources[split][sources[split].template_id.isin(templates[role])]
        if set(complete._manifest_id) != set(partitions[role]):
            raise ValueError(f"WinoBias split breaks template family: {role}")
    if any(
        templates[a] & templates[b]
        for a, b in [("dev", "few_shot"), ("dev", "test"), ("few_shot", "test")]
    ):
        raise ValueError("WinoBias template leakage across partitions")
    dev = sources["dev"].set_index("_manifest_id")
    locations = {}
    for i, block in enumerate(blocks):
        for unit in dev.loc[block].template_id:
            if unit in locations and locations[unit] != i:
                raise ValueError(
                    "WinoBias family split across search/validation blocks"
                )
            locations[unit] = i
