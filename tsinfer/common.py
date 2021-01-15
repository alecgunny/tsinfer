class StreamingMetric:
    """
    Class for updating measurement mean and variance
    statistics in an online fashion using a moving
    (and possibly decaying) average
    Parameters
    ----------
    decay: float or None
        Decay value between 0 and 1 for measurement updates.
        Higher values mean older measurements are downweighted
        more quickly. If left as `None`, a true online average
        will be used
    """

    def __init__(self, decay=None):
        if decay is not None:
            assert 0 < decay and decay <= 1
        self.decay = decay
        self.samples_seen = 0
        self.mean = 0
        self.var = 0

    def update(self, measurement):
        self.samples_seen += 1

        decay = self.decay or 1.0 / self.samples_seen
        delta = measurement - self.mean

        self.mean += decay * delta
        self.var = (1 - decay) * (self.var + decay * delta ** 2)
