from abc import abstractmethod
from collections import defaultdict
import re

import numpy as np

from tsinfer.pipeline.common import StreamingMetric


def pause_context(f):
    def wrapper(self, new):
        with self.simulator.pause():
            self.pause()
            stuff = f(self, new)
            self.initialize_data()
            self.simulator.pipeline.clear_qs()

        self.start_and_warm_up()
        self.resume()
        return stuff
    return wrapper


class VizApp:
    def __init__(
            self,
            simulator,
            data_update_cadence_ms=50,
            plot_update_cadence_ms=200,
            warm_up_batches=50
    ):
        self.simulator = simulator
        self.data_update_cadence_ms = data_update_cadence_ms
        self.plot_update_cadence_ms = plot_update_cadence_ms
        self.warm_up_batches = warm_up_batches
        self.layout = None

        self.build_sources()
        self.initialize_data()

        self._paused = False

    @property
    def paused(self):
        return self._paused

    def pause(self):
        self._pause = True

    def resume(self):
        self._paused = False

    def initialize_data(self):
        self.streaming_metrics = defaultdict(StreamingMetric)
        self.data_streams = defaultdict(lambda : np.array([]))
        self.profile = defaultdict(lambda : defaultdict(StreamingMetric))

        self.start_time = None

    def _get_data(self):
        if not self.paused:
            self.get_data()

    def start_and_warm_up(self):
        if not self.simulator.pipeline.is_alive():
            self.simulator.start()

        print("Warming up for {} batches".format(self.warm_up_batches))
        for _ in range(self.warm_up_batches):
            self.simulator.get()
        self.simulator.pipeline.clear_profile_qs()

    def __call__(self, doc):
        if self.layout is None:
            self.build_layout()
        doc.add_root(self.layout)

        # update our data frequently
        doc.add_periodic_callback(self._get_data, self.data_update_cadence_ms)

        # TODO: should the signal traces be updated
        # at a representative cadence?
        for attr in self.__dir__():
            if re.match("update_.+_plot", attr):
                func = getattr(self, attr)
                doc.add_periodic_callback(func, self.plot_update_cadence_ms)
        self.start_and_warm_up()

    def run(self, server):
        try:
            server.io_loop.add_callback(server.show, "/")
            server.io_loop.start()
        except Exception as e:
            self.simulator.stop()
            raise e

    @abstractmethod
    def build_sources(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def build_layout(self):
        pass
