from abc import abstractmethod
import multiprocessing as mp
import queue
import time


class StoppableIteratingBuffer:
    '''
    Parent class for callable Process targets
    '''
    _LATENCY_WHITELIST = []
    def __init__(self, q_in=None, q_out=None, profile=False):
        self.q_in = q_in
        self.q_out = q_out
        self.param_q = mp.JoinableQueue(1)

        if profile:
            self.profile_q = mp.Queue()
        self.profile = profile

        self._pause_event = mp.Event()
        self._stop_event = mp.Event()

    def put(self, x, timeout=None):
        if self.q_out is None:
            raise ValueError("Nowhere to put!")
        self.q_out.put(x, timeout=timeout)

    def get(self, timeout=None):
        if self.q_in is None:
            raise ValueError("Nowhere to get!")
        try:
            return self.q_in.get(timeout=timeout)
        except queue.Empty as e:
            raise e

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
        try:
            self.initialize_loop()
            while not self.stopped:
                if not self.paused:
                    try:
                        x, y, batch_start_time = self.get_data()
                    except queue.Empty:
                        continue
                    else:
                        self.run(x, y, batch_start_time)
                else:
                    # if paused, try to update parameters
                    try:
                        new_params = self.param_q.get_nowait()
                    except queue.Empty:
                        continue
                    else:
                        params = {
                            param: new_params.get(param, value) for 
                                param, value in self.params.items()
                        }
                        self.initialize(**params)
                        self.param_q.task_done()

        except:
            self.cleanup()
            raise
        else:
            self.cleanup()

    def initialize_loop(self):
        pass

    def get_data(self):
        return self.get(timeout=1e-6)

    @abstractmethod
    def run(self, x, y, batch_start_time):
        '''
        required to have this method for main funcionality
        '''
        pass

    def initialize(self):
        pass

    def cleanup(self):
        pass

    @staticmethod
    def profile(f):
        def wrapper(self, *args, **kwargs):
            if self.profile:
                start_time = time.time()
                stuff = f(self, *args, **kwargs)
                end_time = time.time()

                self.profile_q.put((f.__name__, end_time-start_time))
            else:
                stuff = f(self, *args, **kwargs)
            return stuff
        return wrapper
