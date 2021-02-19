import os
import warnings

import jsonlines
from typing import List


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def write_jsonl_iterable(
    data,
    jsonl_path: str,
    all_keys: List[str] = None,
    insert_placehold: bool = True,
    dumps = None,
):
    if all_keys is not None:
        all_keys_set = set(all_keys)
    else:
        all_keys_set = set()
    jsonl_dir = os.path.dirname(jsonl_path)
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)

    with jsonlines.open(jsonl_path, 'w', dumps=dumps) as fw:
        # align data to make the key follow the same order
        # non existing fields are None
        for d in data:
            if all_keys is None:
                all_keys = list(d.keys())
                all_keys_set = set(all_keys)
                print('all_keys', len(all_keys), all_keys)
            if len(set(d.keys()) - all_keys_set) > 0:
                warnings.warn('Extra keys found and not written to jsonl!')
            d_copy = {}
            for k in all_keys:
                if k in d:
                    d_copy[k] = d[k]
                elif insert_placehold:
                    d_copy[k] = None
            fw.write(d_copy)

