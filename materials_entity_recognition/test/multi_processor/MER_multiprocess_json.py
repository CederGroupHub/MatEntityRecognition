import os

#########################################
# use cpu
#########################################
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pprint import pprint
import itertools
import json
from datetime import datetime
import socket
from typing import List, Any
# https://github.com/leopd/timebudget
from timebudget import timebudget
import shutil
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_shared
import dask.dataframe as dd
import random
import tensorflow as tf

import db_iteration_utils
from db_iteration_utils import exec_mongo_query_tasks
from db_iteration_utils import MongoQueryIndexDocument
from parallelize_utils import run_multiprocessing_tasks_generator
from parallelize_utils import DBQueueWriterHandler
from parallelize_utils import batch_generator
from py_env_utils import found_package

if found_package('materials_entity_recognition'):
    import materials_entity_recognition as MER
if found_package('DataMiningSynthesisLib'):
    from DataMiningSynthesisLib.core import preprocessing


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

ALL_AVAILABLE_GPUS = []

def db_annotate_process(
    queue: mp.Queue,
    save_dir: str,
):
    # db collection is not directly passed as an argument because
    # python can't pickle _thread.lock objects in pymongo objects

    while True:
        data_batch = queue.get()
        if data_batch is None:
            break

        meta_ids, results_batch = data_batch

        i = 0
        total = len(meta_ids)

        for (meta_id, r_para) in zip(meta_ids, results_batch):
            try:
                entry = {
                    'paragraph_id': meta_id,
                    'MER_result': r_para,
                    'MER_version': MER.__version__,
                }

                with open(os.path.join(save_dir, meta_id), 'w') as fw:
                    json.dump(entry, fw, indent=2)
            except Exception as e:
                raise e

            i += 1

def pipeline_in_one_thread(
    tasks_queue: mp.Queue,
    tasks_queue_index: int,
    success_shared: mp_shared.Synchronized,
    device_name: str,
    db_writer_queue: mp.Queue,
    task_name: str,
):
    with tf.device(device_name):
        # =======================================================
        # add initialization part of the pipeline here
        pre_processor = preprocessing.TextPreprocessor('Just for init')
        if os.path.exists(os.path.join('rsc/models')):
            model_dir = os.path.join('rsc/models')
            MER_model = MER.MatRecognition(
                model_path=os.path.join(model_dir, 'matRecognition'),
                bert_path=os.path.join(model_dir, 'MATBert_config'),
                mat_identify_model_path=os.path.join(model_dir, 'matIdentification'),
                mat_identify_bert_path=os.path.join(model_dir, 'Bert_config'),
            )
        else:
            MER_model = MER.MatRecognition()
        # =======================================================

        # success = True
        success_tasks = []
        error_tasks = []

        with timebudget('{} queue {} by {} @ {}'.format(
            task_name, tasks_queue_index, device_name, socket.gethostname())
        ):
            while True:
                records = tasks_queue.get()
                if records is None:
                    break

                if not success_shared.value:
                    continue

                para_batch = []
                meta_ids = []

                for para in records:
                    doc = pre_processor._process(para['text'])
                    para_batch.append(doc.user_data['text'])
                    meta_ids.append(para['_id']['$oid'])

                try:
                    results_batch = MER_model.mat_recognize(para_batch)
                    db_writer_queue.put((meta_ids, results_batch))
                    # if len(success_tasks) > 0:
                    #     raise mp.ProcessError(f'Queue {tasks_queue_index}: Error for debug!')
                except Exception as e:
                    results_batch = []
                    with success_shared.get_lock():
                        success_shared.value = 0
                    print(f'{task_name} queue {tasks_queue_index}: {e}')
                    raise e

                if success_shared.value:
                    success_tasks.extend(meta_ids)
                else:
                    error_tasks.extend(meta_ids)


def run_MER_on_query(
    query: Any,
    save_dir: str,
    task_name: str,
):
    success = True

    batched_query = batch_generator(query, batch_size=1000)

    with DBQueueWriterHandler(
        queue_writer_func=db_annotate_process,
        save_dir=save_dir,
    ) as db_writer_queue:
        # run parallel task
        # the optimal value of num_cores is different for different cluster
        # for lawrencium lr_6, num_cores=4 is better than 1,2,8
        success = run_multiprocessing_tasks_generator(
            tasks_generator=batched_query,
            thread_func=pipeline_in_one_thread,
            func_args=(db_writer_queue, task_name),
            num_cores=1,
            mp_context='spawn',
            queue_maxsize=16,
        )

    return success

def paragraph_generator(
    df: dd.DataFrame,
    start_index: int,
    end_index: int,
    max_num: int
):
    assert len(df) == max_num, (
        'len(df) {} != max_num from iter_index {}'.format(len(df), max_num)
    )
    for (i, dask_row) in itertools.islice(
        zip(range(max_num), df.iterrows()),
        start_index,
        end_index
    ):
        # df index from dask is not consecutive because it reads data by chunk
        (_, record) = dask_row
        record = record.to_dict()
        yield record


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    db_iteration_utils.SAVE_TO = 'generated/mongo_cache/local_iter_index.json'

    # constant parameters
    task_name = 'MER_synthesis_paragraphs'
    save_dir = os.path.abspath('generated/MER_results')
    segment_len = 1000
    step_size = None
    query_cache_path = os.path.join(
        # 'rsc/synthesis_paragraphs_text.jsonl'
        'rsc/test_paragraphs.jsonl'
    )
    source = {
        'index_key': task_name,
    }

    # create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.dirname(db_iteration_utils.SAVE_TO)):
        os.makedirs(os.path.dirname(db_iteration_utils.SAVE_TO))

    # load cached _id's
    df_id_to_query = dd.read_json(
        url_path=query_cache_path,
        blocksize=2**26,
        dtype=object,
    )
    df_id_to_query = df_id_to_query.where(df_id_to_query.notnull(), None)

    # Can be removed in real cases.
    # This is only used for generating more data in this demo.
    df_id_to_query = dd.concat([df_id_to_query]*100, ignore_index=True)

    # generate all tasks
    if MongoQueryIndexDocument.get_iter_index(source=source) is None:
        data_len = len(df_id_to_query)
        print('data_len', data_len)

        # create task indices if not existing
        iter_index = MongoQueryIndexDocument.get_iter_index(
            source=source,
            data_len=data_len,
            segment_len=segment_len,
        )

    print('start to run tasks')

    while True:
        # get iter_index of the tasks
        all_indices = MongoQueryIndexDocument.find_all(save_to=db_iteration_utils.SAVE_TO)
        all_indices = list(filter(
            lambda x: x.start_index < x.boundary_upper,
            all_indices
        ))

        if len(all_indices) == 0:
            print('All tasks completed', datetime.now())
            shutil.make_archive(
                'MER_results',
                'zip',
                save_dir
            )
            shutil.move('MER_results.zip', os.path.dirname(save_dir))
            break

        iter_index = random.choice(all_indices)

        cached_query = paragraph_generator(
            df=df_id_to_query,
            start_index=iter_index.start_index,
            end_index=iter_index.boundary_upper,
            # end_index=min(iter_index.start_index+segment_len, iter_index.boundary_upper) ,
            max_num=iter_index.max_num,
        )

        exec_mongo_query_tasks(
            iter_index=iter_index,
            func_on_query=run_MER_on_query,
            step_size=step_size,
            cached_query=cached_query,
            save_dir=save_dir,
            task_name='{index_key}_{boundary_lower}_{boundary_upper}'
                      ':{start_index}'.format(
                index_key=iter_index.index_key,
                boundary_lower=iter_index.boundary_lower,
                boundary_upper=iter_index.boundary_upper,
                start_index=iter_index.start_index,
            )
        )



