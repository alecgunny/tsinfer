from abc import abstractmethod
import multiprocessing as mp
import queue
import time


class StreamingMetric:
    '''
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
    '''
    def __init__(self, decay=None):
        if decay is not None:
            assert 0 < decay and decay <= 1
        self.decay = decay
        self.samples_seen = 0
        self.mean = 0
        self.var = 0

    def update(self, measurement):
        if self.samples_seen == 0:
            self.mean = measurement
        else:
            decay = self.decay or 1./(self.samples_seen + 1)
            delta = measurement - self.mean
            self.mean += decay*delta
            self.var = (1-decay)*(self.var + decay*delta**2)
        self.samples_seen += 1


def streaming_func_timer(f):
    def wrapper(self, *args, **kwargs):
        if self.profile:
            start_time = time.time()
            stuff = f(self, *args, **kwargs)
            end_time = time.time()

            self.latency_q.put((f.__name__, end_time-start_time))
        else:
            stuff = f(self, *args, **kwargs)
        return stuff
    return wrapper


class StoppableIteratingBuffer:
    '''
    Parent class for callable Process targets
    '''
    _LATENCY_WHITELIST = []
    def __init__(self, q_in=None, q_out=None, profile=False):
        self.q_in = q_in
        self.q_out = q_out

        if profile:
            self.latency_q = mp.Queue()
        self.profile = profile

        self._pause_event = mp.Event()
        self._stop_event = mp.Event()

    def put(self, x, timeout=None):
        if self.q_out is None:
            raise ValueError("Nowhere to put!")
        self.q_out.put(x, timeout=timeout)

    def get(self):
        if self.q_in is None:
            raise ValueError("Nowhere to get!")
        return self.q_in.get()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @property
    def paused(self):
        return self._pause_event.is_set()

    def stop(self):
        self._stop_event.set()

    def pause(self):
        if self._pause_event.is_set():
            raise RuntimeError("Buffer is already paused")
        self._pause_event.set()

    def resume(self):
        if not self._pause_event.is_set():
            raise RuntimeError("Buffer is not paused")
        self._pause_event.clear()

    def __call__(self):
        self.initialize_loop()
        while not self.stopped:
            if not self.paused:
                self.loop()
            else:
                self.check_updates()

        # allow for custom subclass cleanup
        self.cleanup()

    def initialize_loop(self):
        pass

    @abstractmethod
    def loop(self):
        '''
        required to have this method for main funcionality
        '''
        pass

    def check_updates(self):
        pass

    def cleanup(self):
        pass
