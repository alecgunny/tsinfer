import queue
import time

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer



class Postprocessor(StoppableIteratingBuffer):
    __name__ = "Postprocessor"

    def __init__(self, get_postprocess_fn=None, postprocess_kwargs={}, **kwargs):
        # TODO: use partial
        self.get_postprocess_fn = get_postprocess_fn
        self.initialize(postprocess_kwargs)
        super().__init__(**kwargs)

    def initialize(self, postprocess_kwargs):
        if self.get_preprocess_fn is not None:
            self.postprocess_fn = self.get_postprocess_fn(**postprocess_kwargs)
        self.params = {"postprocess_kwargs": postprocess_kwargs}

    @streaming_func_timer
    def postprocess(self, prediction):
        if self.get_postprocess_fn is not None:
            return self.postprocess_fn(prediction)
        return prediction

    def loop(self, x, y, batch_start_time):
        prediction = self.postprocess(x)

        # measure completion time for throughput measurement
        # here to be as accurate as possible
        batch_end_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, y, batch_start_time, batch_end_time))
