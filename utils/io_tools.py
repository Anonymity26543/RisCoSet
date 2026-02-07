import json
import pickle
import gzip
from typing import List, Dict, Iterable
import os
import functools
import hashlib
import numpy as np

def cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs('./outputs/cache', exist_ok=True)
        cache_path = os.path.join('./outputs/cache', hashlib.md5((str(args) + str(kwargs)).encode('utf-8')).hexdigest() + '.pickle')
        if os.path.exists(os.path.join(cache_path)):
            cached_result = Tools.load_pickle(cache_path)
        else:
            cached_result = func(*args, **kwargs)
            os.makedirs('./data/cache', exist_ok=True)
            Tools.dump_pickle(cache_path, cached_result)
        return cached_result
    return wrapper


class Tools:
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        json_list = []
        if file_path.endswith('.gz'):
            with open(file_path, 'rb') as gzfp:
                with gzip.open(gzfp, 'rt') as fp:
                    for line in fp:
                        if any(not c.isspace() for c in line):
                            json_list.append(json.loads(line.strip()))
        else:
            with open(file_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    if any(not c.isspace() for c in line):
                        json_list.append(json.loads(line.strip()))
        return json_list

    @staticmethod
    def stream_jsonl(file_path: str) -> Iterable[Dict]:
        if file_path.endswith('.gz'):
            with open(file_path, 'rb') as gzfp:
                with gzip.open(gzfp, 'rt') as fp:
                    for line in fp:
                        if any(not c.isspace() for c in line):
                            yield json.loads(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    if any(not c.isspace() for c in line):
                        yield json.loads(line.strip())
    
    @staticmethod
    def write_jsonl(file_path: str, json_list: List[Dict], append: bool = False) -> None:
        mode = 'ab' if append else 'wb'
        if file_path.endswith('.gz'):
            with open(file_path, mode) as fp:
                with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                    for line in json_list:
                        gzfp.write((json.dumps(line) + '\n').encode('utf-8'))
        else:
            with open(file_path, mode) as fp:   # 'wb' mode doesn't take an encoding argument
                for line in json_list:
                    fp.write((json.dumps(line) + '\n').encode('utf-8'))

    @staticmethod
    def dump_pickle(file_path: str, content: object) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(content, f)
    
    @staticmethod
    def load_pickle(file_path: str) -> object:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod        
    def get_len_logprobs(logprobs):
        lp_lst = []
        for logprob in logprobs:
            for (_, lp) in logprob.items():
                lp_lst.append(lp.logprob)
        return len(lp_lst)

    @staticmethod
    def get_mean_logprobs(logprobs):
        lp_lst = []
        for logprob in logprobs:
            for (_, lp) in logprob.items():
                lp_lst.append(lp.logprob)
        return sum(lp_lst) / len(lp_lst)

    @staticmethod
    def get_ppl(logprobs):
        return np.exp(-Tools.get_mean_logprobs(logprobs))
    
    @staticmethod
    def get_ppl_list(logprobs):
        return np.exp(-np.mean(logprobs))
