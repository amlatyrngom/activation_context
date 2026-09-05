"""
Slice 1 requires e2e training and reporting to work well.
Slice 1 is complete when:
- An independent review has marked our default configs as standard, without pointless pitfalls that make lose percentage points.
    - This does not include things like the embedding dimensions, but the default lrs, clippings, warmup, etc. That we are not forgetting anything.
    - That's because I am not a hyperparam expert, and don't want to mess this up. So defaults should be very good and standard.
- We have a 1000-train/50-report run passing and displayed in a nice report.
    - The independent reviewer should also read this and make sure it's sensible.
    - With warmups and all.
- The loss trend is as expected.

Slice 2 is multi-vector indexing and retrieval.

Slice 3 will include dataset studying into the mix.
"""


class RetrievalTrainingConfig:
    ... # lrs, schedules, clippings, warmup period (10%), epochs.
    ... # for batches (here I think oom is a risk, so handle worst cases well per gpu; like). It's possible that it should not be a knob, but whatever handles the worst cases gracefully.
    ... # Bog standard: We can't afford bad hyperparams.


class RetrievalTrainingStats:
    pass # Timings, losses, reporting data trend.

class RetrievalTrainer:
    def __init__(
        self,
        harness: object,
    ):
        self.harness = harness

    def train(
        self,
        training_config: RetrievalTrainingConfig,
        retrieval_model: ...,
        retrieval_reporter: ...,
        training_data: list[object],
        reporting_data: list[object],
    ) -> RetrievalTrainingStats:
        pass
