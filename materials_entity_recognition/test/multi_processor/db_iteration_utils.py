import os
import random
import json
import time
import pymodm
from pymodm import MongoModel, fields
from pymongo import IndexModel, ASCENDING, HASHED
from bson import ObjectId
from bson import json_util
from typing import Callable, Union, Any
import math
import urllib.parse


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

SAVE_TO = 'mongo_db'

class IterAssitant(MongoModel):

    index_key = fields.CharField(default=None, required=True, blank=False)
    file_name = fields.CharField(default=None, blank=True)
    summary = fields.DictField(default=None, blank=True)
    boundary_lower = fields.IntegerField(default=None, blank=True)
    boundary_upper = fields.IntegerField(default=None, blank=True)
    max_num = fields.IntegerField(default=None, blank=True)
    # iter_index should fall in [boundary_lower, boundary_upper)
    start_index = fields.IntegerField(default=None, blank=True)
    error_indices = fields.ListField(default=[], blank=True)
    # cursor is next cursor because cursor is both input and output
    cursor = fields.CharField(default=None, blank=True)
    error_cursors = fields.ListField(default=[], blank=True)
    # query is last query because query is input
    query = fields.DictField(default=None, blank=True)
    error_queries = fields.ListField(default=[], blank=True)

    class Meta:
        # https://pymodm.readthedocs.io/en/stable/api/index.html#metadata-attributes
        collection_name = ''
        indexes = [
            IndexModel(keys=[('index_key', HASHED)]),
        ]
        ignore_unknown_fields = True
        final=False

    def save_entry(self, save_to='mongo_db'):
        if save_to == 'mongo_db':
            self.save()
        else:
            if not getattr(self, '_id'):
                setattr(
                    self,
                    '_id',
                    ObjectId(),
                )
            if os.path.exists(save_to):
                with open(save_to, 'r') as fr:
                    all_entries = json.load(fr)
            else:
                all_entries = {}
            all_entries[str(self._id)] = self.to_son()
            with open(save_to, 'w') as fw:
                json.dump(all_entries, fw, indent=2, default=json_util.default)

    def refresh_entry(self, save_to='mongo_db'):
        if save_to == 'mongo_db':
            self.refresh_from_db()
        else:
            if not getattr(self, '_id'):
                return
            if not os.path.exists(save_to):
                return

            with open(save_to, 'r') as fr:
                all_entries = json.load(fr)
            if str(self._id) not in all_entries:
                return
            for k, v in all_entries[str(self._id)].items():
                if k == '_id':
                    continue
                setattr(self, k, v)

    @classmethod
    def find_all(cls, source=None, save_to='mongo_db'):
        if save_to == 'mongo_db':
            return cls.objects.raw(source)
        else:
            if os.path.exists(save_to):
                with open(save_to, 'r') as fr:
                    all_entries = json.load(fr)
                entries_recoveried = []
                for k, v in all_entries.items():
                    del v['_id']
                    entries_recoveried.append(cls(**v, _id=ObjectId(k)))
                return entries_recoveried
            else:
                return []

    @classmethod
    def get_iter_index(
        cls,
        source: dict,
        data_len: int=None,
        data: Any=None,
        segment_len: int=None,
        additional_filter: dict=None,
        **kwargs
    ):
        # TODO: separate this function into two: create and query
        # get data_len
        if (data_len is None) and (data is not None):
            data_len = len(data)

        # get iter_index of the task
        iter_index = list(cls.find_all(source=source, save_to=SAVE_TO))
        if len(iter_index) == 0 and (data_len is not None):
            assert (data_len is not None)
            # get segment_len
            if segment_len is None:
                segment_len = data_len
            # get num_segs
            num_segs = math.ceil(data_len/float(segment_len))
            for i in range(num_segs):
                # create one if no iter_index found
                new_index = cls(
                    index_key=source['index_key'],
                    file_name=source.get('file_name'),
                    summary=source.get('summary'),
                    cursor=kwargs.get('start_cursor'),
                    query=kwargs.get('start_query'),
                    start_index=i*segment_len+kwargs.get('start_index', 0),
                    boundary_lower=i*segment_len+kwargs.get('start_index', 0),
                    boundary_upper=min(
                        (i+1)*segment_len, data_len
                    ) + kwargs.get('start_index', 0),
                    max_num=data_len+kwargs.get('start_index', 0),
                )
                # set extra parameters in sub classes
                self_fields = cls.get_db_fields()
                base_fields = IterAssitant.get_db_fields()
                for f in (set(self_fields) - set(base_fields)):
                    if f not in kwargs:
                        continue
                    setattr(new_index, f, kwargs[f])

                new_index.save_entry(save_to=SAVE_TO)
                iter_index.append(new_index)

        if additional_filter is not None:
            iter_index = list(cls.find_all(
                source={**source, **additional_filter},
                save_to=SAVE_TO
            ))

        if len(iter_index) == 0:
            return None

        # random select one if multiple records found
        iter_index = random.choice(iter_index)

        if (data_len is not None) and (iter_index.max_num < data_len):
            iter_index.max_num = data_len
            iter_index.save_entry(save_to=SAVE_TO)

        return iter_index

    @staticmethod
    def generate_download_url(
        base_url: str,
        basename: Union[int, str],
    ):
        # return example: https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/1971
        if not base_url.endswith('/'):
            base_url += '/'
        url = urllib.parse.urljoin(base_url, str(basename))

        return url

    @staticmethod
    def get_url_basename(url):
        if url.endswith('/'):
            url = url.rstrip('/')
        return os.path.basename(url)


    @classmethod
    def get_db_fields(cls):
        db_fields = []
        for f in dir(cls):
            if isinstance(getattr(cls, f), fields.MongoBaseField):
                db_fields.append(f)
        return db_fields

    @classmethod
    def is_db_connected(cls):
        is_connected = False

        connected_db_names = set([
            x.parsed_uri['database'] for x in pymodm.connection._CONNECTIONS.values()
        ])

        db_name = None
        if 's_db_name' in dir(cls.Meta):
            db_name = cls.Meta.s_db_name

        if db_name is not None:
           if db_name in connected_db_names:
               is_connected = True
        else:
            if len(connected_db_names) > 0:
                is_connected = True

        return is_connected

class MongoQueryIndexDocument(IterAssitant):
    """
        process files from patentsview
    """

    # summary is the same for the same query task
    # all values should be str in summary of MongoQueryIndexDocument
    summary = fields.DictField(default=None, required=True, blank=False)
    # iter_index should in [boundary_lower, boundary_upper)
    start_index = fields.IntegerField(default=0, required=True, blank=False)
    boundary_lower = fields.IntegerField(default=0, required=True, blank=False)
    boundary_upper = fields.IntegerField(default=0, required=True, blank=False)
    max_num = fields.IntegerField(default=0, required=True, blank=False)
    error_indices = fields.ListField(default=[], blank=True)

    class Meta:
        # https://pymodm.readthedocs.io/en/stable/api/index.html#metadata-attributes
        collection_name = 'mongo_query_iter_index'
        indexes = IterAssitant.Meta.indexes
        ignore_unknown_fields = True
        final=True

    def clean(self):
        # Custom validation that requires looking at several fields.
        for v in self.summary.values():
            if not isinstance(v, str):
                raise pymodm.errors.ValidationError(
                    'Value in MongoQueryIndexDocument.summary should be str '
                    'instead of {}!'.format(type(v))
                )


def exec_mongo_query_tasks(
    iter_index: MongoQueryIndexDocument,
    func_on_query: Callable,
    step_size: int = None,
    cached_query: Any=None,
    **func_kwargs
):
    assert isinstance(iter_index, MongoQueryIndexDocument)

    print(
        'Task {index_key}_{boundary_lower}_{boundary_upper} started: '
        '{start_index}'.format(
            index_key=iter_index.index_key,
            boundary_lower=iter_index.boundary_lower,
            boundary_upper=iter_index.boundary_upper,
            start_index=iter_index.start_index,
        )
    )

    # container to record errors
    error_indices = []

    if iter_index.boundary_upper <= iter_index.start_index:
        return

    # get start_index and end_index
    start_index = iter_index.start_index
    if step_size is None:
        end_index = iter_index.boundary_upper
    else:
        end_index = start_index + step_size

    if cached_query is None:
        db = pymodm.connection._get_db()
        col = db[iter_index.summary['col_name']]
        query = col.find(
            filter=json.loads(iter_index.summary['filter']),
            projection=(
                json.loads(iter_index.summary['projection'])
                if 'projection' in iter_index.summary else None
            ),
            skip=start_index,
            limit=end_index-start_index,
            no_cursor_timeout=True,
        )
    else:
        query = cached_query

    # start tasks
    success = func_on_query(
        query=query,
        **func_kwargs
    )
    if not success:
        for i in range(start_index, end_index):
            error_indices.append(i)

    # update iter_index
    iter_index.refresh_entry(save_to=SAVE_TO)
    if len(error_indices) > 0:
        iter_index.error_indices += error_indices
        iter_index.save_entry(save_to=SAVE_TO)
    if iter_index.start_index < end_index:
        iter_index.start_index = end_index
        iter_index.save_entry(save_to=SAVE_TO)

    print(
        '{num_entries} entries from {index_key}_{boundary_lower}_{boundary_upper} '
        'stored! Progress: {progress}'.format(
            num_entries=(iter_index.start_index - start_index),
            index_key=iter_index.index_key,
            boundary_lower=iter_index.boundary_lower,
            boundary_upper=iter_index.boundary_upper,
            progress=(
                 iter_index.start_index-iter_index.boundary_lower
            )/max(iter_index.boundary_upper-iter_index.boundary_lower, 1)
        )
    )


if __name__ == '__main__':
    # https://github.com/leopd/timebudget
    from timebudget import timebudget
    import pymodm
    import urllib.parse

    with open('config.json', 'r') as fr:
       config = json.load(fr)

    # connect to db
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

    files = list(MongoQueryIndexDocument.objects.raw({

    }))
    file_names = [str(f.summary) for f in files]
    print('len(file_names)', len(file_names))
    print('len(set(file_names))', len(set(file_names)))
    with open('generated/file_url.json', 'w') as fw:
        json.dump(file_names, fw, indent=2)
