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


def dense_candidate_class():
    """Realizable 49-model class with coarse and nearby alternatives."""
    truth = np.asarray(TRUE_WEIGHTS)
    models = list(candidate_class()[:-1])
    for coordinate in range(len(truth)):
        for shift in (-.15, .15):
            weights = truth.copy()
            weights[coordinate] = np.clip(weights[coordinate] + shift, .05, .95)
            coarse = models[2*coordinate + int(shift > 0)]
            if tuple(weights) == coarse.weights:
                weights[coordinate] = (truth[coordinate] + coarse.weights[coordinate]) / 2
            models.append(QueueModel(tuple(weights)))
    models.append(QueueModel(tuple(truth)))
    if len({model.weights for model in models}) != len(models):
        raise RuntimeError("dense candidate class contains duplicate models")
    return tuple(models)


def fine_candidate_class():
    """Realizable 73-model class with a third, closer parameter scale."""
    truth = np.asarray(TRUE_WEIGHTS)
    models = list(dense_candidate_class()[:-1])
    for coordinate in range(len(truth)):
        for shift in (-.05, .05):
            weights = truth.copy()
            weights[coordinate] = np.clip(weights[coordinate] + shift, .05, .95)
            models.append(QueueModel(tuple(weights)))
    models.append(QueueModel(tuple(truth)))
    if len({model.weights for model in models}) != len(models):
        raise RuntimeError("fine candidate class contains duplicate models")
    return tuple(models)


def finite_mle(models, counts):
    if not counts:
        return 0
    scores = np.zeros(len(models))
    for trace, count in counts.items():
        scores += count * np.log([model.trajectory_probability(trace)
                                  for model in models])
    return int(scores.argmax())
