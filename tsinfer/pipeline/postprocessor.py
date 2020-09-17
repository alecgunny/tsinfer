import multiprocessing as mp
import queue
import time

from tsinfer.pipeline.common import StoppableIteratingBuffer, streaming_func_timer



class Postprocessor(StoppableIteratingBuffer):
    __name__ = "Postprocessor"

    def __init__(self, get_postprocess_fn, postprocess_kwargs={}, **kwargs):
        self.get_postprocess_fn = get_postprocess_fn
        self.postprocess_kwargs = postprocess_kwargs
        self.postprocess_fn = get_postprocess_fn(**postprocess_kwargs)

        self.param_q = mp.JoinableQueue(1)
        super().__init__(**kwargs)

    @streaming_func_timer
    def postprocess(self, prediction):
        if self.postprocess_fn is not None:
            return self.postprocess_fn(prediction)
        return prediction

    def check_updates(self):
        try:
            new_postprocess_kwargs = self.param_q.get_nowait()
        except queue.Empty:
            return
        else:
            print("updating too")
            self.postprocess_kwargs.update(new_postprocess_kwargs)
            self.postprocess_fn = self.get_postprocess_fn(**self.postprocess_kwargs)

            self.param_q.task_done()

    def loop(self):
        while True:
            try:
                prediction, target, batch_start_time = self.get(timeout=1e-6)
                break
            except queue.Empty:
                if self.paused:
                    return
        prediction = self.postprocess(prediction)

        # measure completion time for throughput measurement
        # here to be as accurate as possible
        batch_end_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, target, batch_start_time, batch_end_time))

