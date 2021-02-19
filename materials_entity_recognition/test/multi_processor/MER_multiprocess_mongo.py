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
import pymodm
import urllib.parse
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_shared
import pymongo
from bson import json_util
from bson import ObjectId
import dask.dataframe as dd
from pymongo import HASHED, ASCENDING
import random
import tensorflow as tf

from db_iteration_utils import exec_mongo_query_tasks
from db_iteration_utils import MongoQueryIndexDocument

from parallelize_utils import run_multiprocessing_tasks_generator
from parallelize_utils import DBQueueWriterHandler
from parallelize_utils import batch_generator
from json_utils import write_jsonl_iterable
from mongo_utils import get_mongo_db
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
    mongo_config: dict,
    save_col_name: str,
):
    # db collection is not directly passed as an argument because
    # python can't pickle _thread.lock objects in pymongo objects
    db = get_mongo_db(mongo_config=mongo_config)
    db_col = db[save_col_name]
    db_col.create_index([('paragraph_id', HASHED)], unique=False)
    db_col.create_index([('MER_version', ASCENDING)], unique=False)

    while True:
        data_batch = queue.get()
        if data_batch is None:
            break

        meta_ids, results_batch = data_batch

        i = 0
        total = len(meta_ids)

        for (meta_id, r_para) in zip(meta_ids, results_batch):
            try:
                db_col.update_one(
                    {'paragraph_id': meta_id},
                    {
                        '$set': {
                            'MER_result': r_para,
                            'MER_version': MER.__version__,
                        }
                    },
                    upsert=True,
                )
            except pymongo.errors.DocumentTooLarge as e:
                print('pymongo.errors.DocumentTooLarge', meta_id)
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
                    meta_ids.append(para['_id'])

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
    mongo_config: dict,
    save_col_name: str,
    task_name: str,
):
    success = True

    batched_query = batch_generator(query, batch_size=1000)

    with DBQueueWriterHandler(
        queue_writer_func=db_annotate_process,
        mongo_config=mongo_config,
        save_col_name=save_col_name,
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
    col_query: pymongo.collection.Collection,
    df:  dd.DataFrame,
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
        para = col_query.find_one(
            {'_id': ObjectId(record['paragraph_id']['$oid'])}
        )
        yield para


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # conneceting to database

    # always get db handler from main
    with open(os.path.join('config.json'), 'r') as fr:
       config = json.load(fr)

    # connect to data fetching db
    db_fetch = get_mongo_db(mongo_config=config['mongo_db_fetch'])

    # connect to data storing db
    pymodm.connect(
        mongodb_uri=(
            'mongodb://{username}:{password}@{host}:{port}/{db}?'
            'authSource={authentication_source}'.format(
                host=config['mongo_db']['host'],
                port=config['mongo_db']['port'],
                username=urllib.parse.quote_plus(config['mongo_db']['username'], safe=''),
                password=urllib.parse.quote_plus(config['mongo_db']['password'], safe=''),
                db=config['mongo_db']['db_name'],
                authentication_source=config['mongo_db']['auth_source'],
            )
        )
    )

    # db_store = pymodm.connection._get_db()
    db_store = get_mongo_db(mongo_config=config['mongo_db'])

    # order of key is reserved in newest python
    # it is actually ordered dict
    task_name = 'MER_synthesis_paragraphs'
    paragraph_col_name = 'Paragraphs'
    query_col_name = 'Paragraphs_Meta'
    query_filter_str= json.dumps({
        'classification': {
            '$exists': True,
            '$nin': ['something_else', None],
        }
    })
    query_projection = {
        'DOI': True,
        'paragraph_id': True,
    }

    segment_len = 10000
    step_size = None
    query_cache_path = os.path.join(
        'generated/mongo_cache/synthesis_paragraphs.jsonl'
    )

    # get query parameters
    source = {
        'index_key': task_name,
        'summary': {
            'col_name': query_col_name,
            'filter': query_filter_str,
        },
    }

    # cache _id's of example sections
    if not os.path.exists(query_cache_path):
        col = db_fetch[query_col_name]
        query = col.find(
            filter=json.loads(query_filter_str),
            projection=query_projection,
        )
        write_jsonl_iterable(
            data=query,
            jsonl_path=query_cache_path,
            dumps=json_util.dumps,
        )

    # load cached _id's
    df_id_to_query = dd.read_json(
        url_path=query_cache_path,
        blocksize=2**26,
        dtype=object,
    )
    df_id_to_query = df_id_to_query.where(df_id_to_query.notnull(), None)

    # generate iter_index
    if MongoQueryIndexDocument.get_iter_index(source) is None:
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
        iter_index = MongoQueryIndexDocument.get_iter_index(
            source=source,
            additional_filter={
                '$where': 'this.start_index < this.boundary_upper'
            }
        )
        if iter_index is None:
            print('All tasks completed', datetime.now())
            break

        cached_query = paragraph_generator(
            col_query=db_fetch[paragraph_col_name],
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
            mongo_config=config['mongo_db'],
            save_col_name='synthesis_paragraphs_meta',
            task_name='{index_key}_{boundary_lower}_{boundary_upper}'
                      ':{start_index}'.format(
                index_key=iter_index.index_key,
                boundary_lower=iter_index.boundary_lower,
                boundary_upper=iter_index.boundary_upper,
                start_index=iter_index.start_index,
            )
        )


