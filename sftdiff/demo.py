"""Small synthetic fixtures, explicitly not a real training benchmark."""

import copy

from .capture import capture_batch
from .snapshot import load_snapshot


class DemoTokenizer:
    name_or_path = "sftdiff/synthetic-demo-v1"
    all_special_ids = [0, 1, 5, 7, 8, 9]
    pieces = ["<bos>", "<user>", "Name", "a", "color", "<assistant>", "Blue", "<eos>", "<pad>", "<tool>", "Stock", "42"]

    def get_vocab(self):
        return {token: index for index, token in enumerate(self.pieces)}

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[token] for token in ids]


def create_demo(before_path, after_path, *, force=False):
    ids = [0, 1, 2, 3, 4, 5, 6, 7]
    labels = [-100] * 6 + [6, 7]
    names = ["order-001-unchanged", "order-002-mask-drift", "order-003-answer-truncated",
             "order-004-eos-removed", "order-005-padding", "order-006-removed"]
    before = {"input_ids": [ids.copy() for _ in names],
              "labels": [labels.copy() for _ in names],
              "attention_mask": [[1] * len(ids) for _ in names]}
    before["input_ids"][4] += [8, 8]
    before["labels"][4] += [-100, -100]
    before["attention_mask"][4] += [0, 0]
    after = copy.deepcopy(before)
    after["labels"][1][2:5] = ids[2:5]
    for key in after:
        after[key][2] = after[key][2][:5]
        after[key][3] = after[key][3][:-1]
    after["labels"][4][-2:] = [8, 8]
    after_names = names[:-1] + ["order-007-added"]
    common = {"synthetic": True, "note": "Hand-authored token evidence, not measured model output.",
              "packing": False}
    capture_batch(before_path, before, DemoTokenizer(), row_ids=names,
                  metadata={**common, "pipeline": "baseline fixture"}, force=force)
    capture_batch(after_path, after, DemoTokenizer(), row_ids=after_names,
                  metadata={**common, "pipeline": "changed fixture"}, force=force)
    return load_snapshot(before_path), load_snapshot(after_path)
