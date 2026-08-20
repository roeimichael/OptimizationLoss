"""When to stop the CE pass during the constraint phase.

THE reason this exists, measured on ViTB16 x dermmnist x L30_G30 (2026-08-20):

Every constraint epoch runs the full CE pass first -- 126 optimizer steps at
batch_size 64 over 8,012 training images -- and then takes exactly ONE
constraint step, which is additionally capped at unit norm. Over 29 epochs that
is 3,654 CE updates against 29 clipped constraint updates.

The result is not that the constraint is weak. It is that the constraint is
outvoted. TraLO's hard count on the capped class went 125 -> 121 -> 251 -> 250
-> 353 -> 205 -> 281 across the phase, against a budget of 67: it STARTED at
1.9x the budget and ENDED at 4.2x. CE is driving the melanoma count up toward
its true prevalence (223 of 2,003) faster than one clipped step per epoch can
pull it down, and `Global_Satisfied` was 0 on every logged epoch.

An older version of this project turned CE off once training accuracy
saturated, and TraLO satisfied its constraints. That mechanism was deleted, and
the deletion was RIGHT for the reason given -- `enable_ce_skip` was declared by
TraLO alone, so a campaign ran the gate off for TraLO and on for both duals,
fabricating a 0.22 cc-F1 artifact against a 0.019-0.031 margin. It was removed
for being ASYMMETRIC. It was never measured as harmful when applied fairly.

So it lives here, as one object every trained arm constructs from the SHARED
`constraint_phase` block, rather than as a per-arm hyperparameter any single
methodology can declare on its own. That is the structural fix for the defect
that got it deleted: there is no per-arm key to set differently.

Gating on SATURATION rather than switching CE off from the start is what should
protect quality -- the CE objective has already converged by the time the gate
fires, so the remaining epochs spend budget on the constraint instead of
re-optimising something that has stopped moving. The failure mode to watch is
the `joint` arm's: it held the cap on 98.8% of epochs by crushing the count to
0.26K during training and lost 0.067 AP doing it. Satisfaction bought by
wrecking the classifier is not the goal; macro-F1 against `clip` is the check.

Disabled by default (`ce_skip_acc: 0.0`), so nothing changes until a protocol
sets it.
"""

import logging

log = logging.getLogger(__name__)


class CESaturationSkip:
    """Stop the CE pass after train accuracy holds above a threshold.

    Construct from the arm's resolved hyperparameters. Both keys come from the
    shared `constraint_phase` block, so every trained arm in a campaign gets
    the same schedule or none of them do.
    """

    def __init__(self, hp):
        self.threshold = float(hp.get("ce_skip_acc", 0.0) or 0.0)
        self.patience = int(hp.get("ce_skip_patience", 2) or 2)
        self._streak = 0
        self.skipping = False
        self.skip_from_epoch = None

    @property
    def enabled(self):
        """A threshold of 0.0 means the gate is off -- CE runs every epoch."""
        return self.threshold > 0.0

    def should_skip(self):
        return self.skipping

    def update(self, train_acc, epoch):
        """Call once per epoch, AFTER the CE pass, with that epoch's accuracy.

        Latches: once the gate fires it stays fired. A model that dips back
        below the threshold for one epoch has not become un-converged, and an
        un-latching gate would reintroduce the CE force it exists to remove.
        """
        if not self.enabled or self.skipping:
            return
        if train_acc is None:
            return
        if train_acc >= self.threshold:
            self._streak += 1
            if self._streak >= self.patience:
                self.skipping = True
                self.skip_from_epoch = epoch
                log.info(
                    "CE-skip: train_acc held >= %.3f for %d consecutive epochs; "
                    "stopping the CE pass from epoch %d so the remaining "
                    "constraint epochs are not outvoted 126:1.",
                    self.threshold, self.patience, epoch)
        else:
            self._streak = 0

    def summary(self):
        return {
            "ce_skip_enabled": self.enabled,
            "ce_skip_acc": self.threshold,
            "ce_skip_patience": self.patience,
            "ce_skip_fired": self.skipping,
            "ce_skip_from_epoch": self.skip_from_epoch,
        }
