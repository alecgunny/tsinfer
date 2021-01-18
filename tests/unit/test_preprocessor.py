import queue
import sys
import time
from multiprocessing import Process, Queue

from tsinfer.pipeline import DummyDataGenerator, Path, Preprocessor


def _stop_process(p, join_timeout=0.5):
    p.join(join_timeout)
    if p.exitcode:
        raise RuntimeError(f"Process {p.name} exited with code {p.exitcode}")

    try:
        p.close()
        return True
    except ValueError:
        p.terminate()
        time.sleep(0.1)
        p.close()
        return False


def _run_preprocessor(q, num_steps=100):
    values = []
    for _ in range(num_steps):
        try:
            x = q.get(timeout=1)
        except queue.Empty:
            return ValueError("Nothing in out q")
        except Exception as e:
            return e

        if isinstance(x, Exception):
            return x
        values.append(x)
    return values


def test_preprocessor(num_channels=21, qsize=100):
    in_q = Queue(qsize)
    out_q = Queue(qsize)

    data_gen = DummyDataGenerator(num_channels, in_q)
    path = Path(channels=num_channels, q=in_q)

    preprocessor = Preprocessor(
        path,
        batch_size=8,
        kernel_size=1.0,
        kernel_stride=0.002,
        fs=4000,
        q_out=out_q,
    )

    data_gen_p = Process(target=data_gen, name="data_gen")
    data_gen_p.start()
    preproc_p = Process(target=preprocessor, name="preprocessor")
    preproc_p.start()

    values = _run_preprocessor(preprocessor.q_out)

    for target in [data_gen, preprocessor]:
        target.stop()
        assert target.stopped

    ungraceful = []
    for p in [data_gen_p, preproc_p]:
        if not _stop_process(p):
            ungraceful.append(p.name)

    if isinstance(values, Exception):
        raise RuntimeError from values

    # if len(ungraceful) > 0:
    #     raise ValueError(
    #         "Processes {} did not exit gracefully".format(
    #             ", ".join(ungraceful)
    #         )
    #     )
    data_gen = DummyDataGenerator(num_channels, in_q)
    in_q_2 = Queue(qsize)
    data_gen_2 = DummyDataGenerator(num_channels, in_q_2)
    path_2 = Path(channels=num_channels, q=in_q_2)
    preprocessor = Preprocessor(
        {"left": path, "right": path_2},
        batch_size=8,
        kernel_size=1,
        kernel_stride=0.002,
        fs=4000,
        q_out=out_q,
    )

    data_gen_p = Process(target=data_gen, name="data_gen")
    data_gen_2_p = Process(target=data_gen_2, name="data_gen_2")
    preproc_p = Process(target=preprocessor, name="preprocessor")
    for p in [data_gen_p, data_gen_2_p, preproc_p]:
        p.start()

    values = _run_preprocessor(preprocessor.q_out)
    for target in [data_gen, data_gen_2, preprocessor]:
        target.stop()
        assert target.stopped

    ungraceful = []
    for p in [data_gen_p, data_gen_2_p, preproc_p]:
        if not _stop_process(p):
            ungraceful.append(p.name)
    if isinstance(values, Exception):
        raise RuntimeError from values

    if len(ungraceful) > 0:
        raise ValueError(
            "Processes {} did not exit gracefully".format(", ".join(ungraceful))
        )


if __name__ == "__main__":
    test_preprocessor()
