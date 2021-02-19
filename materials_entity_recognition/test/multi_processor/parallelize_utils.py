import math
import os
import time
from pprint import pprint
import ctypes
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import itertools
import tensorflow as tf

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def run_multiprocessing_tasks_generator(
    tasks_generator,
    thread_func,
    func_args=(),
    num_cores=4,
    mp_context=None,
    queue_maxsize=16,
):
    """

    :param tasks_generator:
    :param thread_func: When using MP, each thread_func should handle a queue
        ending with None. When using threading, it does not matter because
        the original tasks_generator will be used
    :param func_args:
    :param num_cores:
    :param verbose:
    :param join_results:
    :param use_threading:
    :return:
    """

    def queue_tasks(queues, tasks_generator):
        for i, items in enumerate(tasks_generator):
            queues[i%num_cores].put(items)
        for q in queues:
            # put None as the ending signal of a queue
            q.put(None)

    # execute pipeline in a parallel way
    mp_ctx = mp.get_context(mp_context)
    queues = []
    subprocesses = []
    success = mp_ctx.Value('i', 1)
    ALL_AVAILABLE_GPUS = tf.config.experimental.list_logical_devices('GPU')
    print('ALL_AVAILABLE_GPUS', ALL_AVAILABLE_GPUS)
    for i in range(num_cores):
        new_queue = mp_ctx.Queue(maxsize=queue_maxsize)
        queues.append(new_queue)
        if len(ALL_AVAILABLE_GPUS) == 0:
            device_name = '/CPU:0'
        else:
            device_name = (ALL_AVAILABLE_GPUS.pop()).name
        # The main reason starmap is not used here is starmap would call list(iter) first,
        # which is not a good implementation for generator consumer.
        subprocesses.append(mp_ctx.Process(
            target=thread_func,
            args=(new_queue, i, success, device_name) + func_args,
        ))

    [process.start() for process in subprocesses]
    queue_tasks(queues, tasks_generator)
    for process in subprocesses:
        process.join()

    print('run_multiprocessing_tasks_generator', success)

    return bool(success.value)


class DBQueueWriterHandler(object):
    def __init__(self,
                 queue_writer_func,
                 mongo_config: dict = None,
                 save_col_name: str = None,
                 save_dir: str = None,
                 ):
        self.mp_ctx = mp.get_context('spawn')  # To be compatible with classifier workers

        self.db_writer_queue = self.mp_ctx.Queue(maxsize=512)
        # db collection is not directly passed as an argument because
        # python can't pickle _thread.lock objects in pymongo objects
        if mongo_config is not None:
            self.process = self.mp_ctx.Process(
                target=queue_writer_func,
                args=(self.db_writer_queue, mongo_config, save_col_name)
            )
        else:
            self.process = self.mp_ctx.Process(
                target=queue_writer_func,
                args=(self.db_writer_queue, save_dir)
            )
        self.process.start()

    def __enter__(self):
        return self.db_writer_queue

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db_writer_queue.put(None)
        self.process.join()


def batch_generator(iterable, batch_size=1):
    current_batch = []
    for i, item in enumerate(iterable):
        current_batch.append(item)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


if __name__ == '__main__':
    for x in batch_generator(range(0, 10), 3):
        print(x)