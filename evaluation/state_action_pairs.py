"""Evaluate opposite English actions at annotated candidate-window centers."""

import numpy as np


ACTION_PAIRS = (
    (("person_picks_up_object", "a person picking up an object"),
     ("person_puts_down_object", "a person putting down an object")),
    (("person_sits_down", "a person sitting down"),
     ("person_stands_up", "a person standing up")),
    (("person_opens_facility_door", "a person opening a door"),
     ("person_closes_facility_door", "a person closing a door")),
    (("person_enters_scene_through_structure", "a person entering the scene through a doorway"),
     ("person_exits_scene_through_structure", "a person leaving the scene through a doorway")),
)


def evaluate_pairs(windows, targets, logits):
    activities = [t for t in targets if t["target_type"] == "activity" and t["evaluable"]]
    active = [{t["activity_name"] for t in activities
               if w["video_id"] == t["video_id"] and t["start_ms"] <= w["timestamp_ms"] < t["end_ms"]}
              for w in windows]
    results = []
    for pair_index, pair in enumerate(ACTION_PAIRS):
        classes = [c for c, _ in pair]
        eligible = [i for i, labels in enumerate(active) if len(labels.intersection(classes)) == 1]
        truth = np.array([classes[1] in active[i] for i in eligible], dtype=bool)
        stats = {}
        for order, scores in logits.items():
            sub = scores[eligible, pair_index * 2 : pair_index * 2 + 2]
            margin = np.where(truth, sub[:, 1] - sub[:, 0], sub[:, 0] - sub[:, 1])
            credit = (margin > 0).astype(float) + 0.5 * (margin == 0)
            recalls = [float(credit[truth == pos].mean()) if np.any(truth == pos) else None
                       for pos in (False, True)]
            stats[order] = dict(credit=float(credit.sum()), ties=int((margin == 0).sum()),
                                recalls=recalls, balanced_accuracy=None if None in recalls else sum(recalls) / 2)
        results.append(dict(classes=classes, prompts=[p for _, p in pair], indices=eligible,
                            true_class_indices=truth.astype(int).tolist(), counts=[int((~truth).sum()), int(truth.sum())],
                            simultaneous=sum(set(classes) <= observed for observed in active), metrics=stats))
    return results
