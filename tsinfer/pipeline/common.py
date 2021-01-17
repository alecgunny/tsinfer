import multiprocessing as mp
import queue
import time
import typing
from abc import abstractmethod

import attr


class StoppableIteratingBuffer:
    """
    Parent class for callable Process targets
    """

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

    def put(self, x, name=None, timeout=None):
        if self.q_out is None:
            raise ValueError("Nowhere to put!")
        elif isinstance(self.q_out, dict):
            if name is None:
                raise ValueError("Must specify q to put!")
            try:
                q_out = self.q_out[name]
            except KeyError:
                raise ValueError(f"Unrecognized path q {name}")
        else:
            q_out = self.q_out
        q_out.put(x, timeout=timeout)

    def get(self, name=None, timeout=None):
        if self.q_in is None:
            raise ValueError("Nowhere to get!")
        elif isinstance(self.q_in, dict):
            if name is None:
                raise ValueError("Must specify q to get!")
            try:
                q_in = self.q_in[name]
            except KeyError:
                raise ValueError(f"Unrecognized path q {name}")
        else:
            q_in = self.q_in
        try:
            return q_in.get(timeout=timeout)
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
                if self.paused:
                    # if paused, try to update parameters
                    try:
                        new_params = self.param_q.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        params = {
                            param: new_params.get(param, value)
                            for param, value in self.params.items()
                        }
                        self.initialize(**params)
                        self.param_q.task_done()
                    continue

                # otherwise, try to do get data and run main
                # process on it
                try:
                    stuff = self.get_data()
                except queue.Empty:
                    continue
                else:
                    # TODO: multi-thread this?
                    for name, (x, batch_start_time) in stuff.items():
                        self.run(x, batch_start_time, name=name)

        except Exception as e:
            self.put(e)
        finally:
            self.cleanup()

    def initialize_loop(self):
        pass

    def get_data(self):
        stuff = {}
        if isinstance(self.q_in, dict):
            for name, q in self.q_in.items():
                try:
                    self.get(name, timeout=1e-6)
                except queue.Empty:
                    continue

            # nothing is running, so raise Empty so
            # that __call__ loop continues
            if len(stuff) == 0:
                raise queue.Empty

            # only some of the queues not providing data,
            # which may mean that one is broken. Give
            # them another chance to provide data, but
            # if they still have nothing then raise an
            # error
            if len(stuff) < len(self.q_in):
                for key in set(self.q_in) - set(stuff):
                    try:
                        self.get(name, timeout=1e-6)
                    except queue.Empty:
                        raise RuntimeError(
                            "Process {} not providing any "
                            "new data".format(name)
                        )
        else:
            stuff[None] = self.get(timeout=1e-6)

        # if any of our input processes raised an error,
        # then pass it along and stop this process. Throw
        # a queue.Empty so that the __call__ process goes
        # to the top of the loop then breaks
        exceptions = list(
            filter(lambda x: isinstance(x, Exception), stuff.values())
        )
        if len(exceptions) > 0:
            self.put(exceptions[0])
            self.stop()
            raise queue.Empty

        return stuff

    @abstractmethod
    def run(self, x, batch_start_time, name=None):
        """
        required to have this method for main funcionality
        """
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

                self.profile_q.put((f.__name__, end_time - start_time))
            else:
                stuff = f(self, *args, **kwargs)
            return stuff

        return wrapper


@attr.s(auto_attribs=True)
class Path:
    channels: typing.Union[typing.List[str], int]
    q: mp.Queue
    processing_fn: typing.Optional[typing.Callable] = None
    processing_fn_kwargs: typing.Optional[dict] = None
