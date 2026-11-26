"""PSR core-action exploration policies for finite-class reward-free OMLE."""


class CoreInterventionPolicy:
    def __init__(self, diagnostic, intervention_stage, fixed_core_actions, rng):
        self.diagnostic = diagnostic
        self.intervention_stage = intervention_stage
        self.fixed_core_actions = fixed_core_actions
        self.rng = rng

    def __getitem__(self, history):
        stage = (len(history) - 1) // 2
        if stage < self.intervention_stage:
            return self.diagnostic[history]
        if stage == self.intervention_stage:
            return self.rng.randrange(2)
        if stage <= self.intervention_stage + self.fixed_core_actions:
            return 0
        return self.rng.randrange(2)


def core_interventions(diagnostic, model, rng):
    """Eq. (10): roll in, random action, next-core actions, random finish."""
    for stage in range(model.horizon - 1):
        core_actions = model.core_length(stage + 1) - 1
        yield CoreInterventionPolicy(diagnostic, stage, core_actions, rng)
