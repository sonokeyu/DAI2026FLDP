"""Shared finite realizable model class and ordinary complete-episode MLE."""
import numpy as np

from queue_pomdp import QueueModel, TRUE_WEIGHTS


def candidate_class():
    truth = np.asarray(TRUE_WEIGHTS)
    models = []
    for index in range(12):
        for shift in (-.6, .6):
            candidate = truth.copy()
            candidate[index] = np.clip(candidate[index] + shift, .05, .95)
            models.append(QueueModel(tuple(candidate)))
    models.append(QueueModel(tuple(truth)))
    return tuple(models)


def finite_mle(models, counts):
    if not counts:
        return 0
    scores = np.zeros(len(models))
    for trace, count in counts.items():
        scores += count * np.log([model.trajectory_probability(trace)
                                  for model in models])
    return int(scores.argmax())
