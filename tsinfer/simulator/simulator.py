import multiprocessing as mp
from contextlib import contextmanager


class Simulator:
    def __init__(self, data_generator, pipeline):
        self.data_generator = data_generator
        self.data_gen_process = mp.Process(target=self.data_generator)

        self.pipeline = pipeline

    def start(self):
        self.pipeline.start()
        self.data_gen_process.start()

    @contextmanager
    def pause(self):
        self.pipeline.pause()
        self.data_generator.pause()

        try:
            yield None
        finally:
            print("Resuming")
            self.pipeline.resume()
            self.data_generator.resume()

    def stop(self):
        self.pipeline.stop()
        self.data_generator.stop()

        for p in self.pipeline.processes:
            if p.is_alive():
                p.join()
        if self.data_gen_process.is_alive():
            self.data_gen_process.join()

        self.clear_qs()

    def get(self):
        return self.pipeline.get()

    def clear_qs(self):
        self.pipeline.clear_qs()
        self.data_generator.q_out.queue.clear()
