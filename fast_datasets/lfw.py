# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/lfw.ipynb.

# %% auto 0
__all__ = ['LFWPeople', 'LFWPairs', 'SLLFWPairs']

# %% ../nbs/lfw.ipynb 2
from abc import ABC, abstractmethod

from fastai.vision.all import *
from sklearn.model_selection import KFold
from fastdownload import FastDownload

import fast_datasets.patches
from .utils import return_list

# %% ../nbs/lfw.ipynb 3
class LFW(ABC):
    BASE_URL = 'http://vis-www.cs.umass.edu/lfw'
    TEST_ITEMS_FILE_NAME: str

    def __init__(self):
        self.root = untar_data(self._url('lfw.tgz'))

    @classmethod
    def _url(cls, fname):
        return f'{cls.BASE_URL}/{fname}'

    def test(self):
        items = self._parse_items(self.TEST_ITEMS_FILE_NAME)
        splits = KFold(n_splits=10, shuffle=False).split(range_of(items))
        return [self._load(items=items, splits=s) for s in splits]

    def _fetch_file(self, fname):
        return FastDownload().download(self._url(fname))

    @abstractmethod
    def _parse_items(self, fname):
        pass

    @abstractmethod
    def _load(self, **kwargs):
        pass

    def _get_path(self, name, num) -> Path:
        return self.root / name / f'{name}_{num:04d}.jpg'


class LFWDevMixin(LFW):
    DEV_TRAIN_ITEMS_FILE_NAME: str
    DEV_TEST_ITEMS_FILE_NAME: str

    def dev(self):
        train_items = self._parse_items(self.DEV_TRAIN_ITEMS_FILE_NAME)
        valid_items = self._parse_items(self.DEV_TEST_ITEMS_FILE_NAME)
        items = valid_items+train_items

        return self._load(
            items=items,
            splits=IndexSplitter(range_of(valid_items))(items)
        )

# %% ../nbs/lfw.ipynb 4
class LFWPeople(LFWDevMixin, LFW):
    """
    Individual facial images.
    Splits contain disjoint identities, since they're meant to for constructing pairs (using `Pairs`)
    """
    TEST_ITEMS_FILE_NAME = 'people.txt'
    DEV_TRAIN_ITEMS_FILE_NAME = 'peopleDevTrain.txt'
    DEV_TEST_ITEMS_FILE_NAME = 'peopleDevTest.txt'
    
    @return_list
    def _parse_items(self, fname):
        lines = [l.split() for l in self._fetch_file(fname).readlines()]
        for l in lines[1:]:
            if len(l) == 1:
                continue
            name, num_images = l
            for i in range(1, int(num_images)+1):
                yield self._get_path(name, i)

    def _load(self, **kwargs):
        return Datasets(
            tfms=[
                PILImage.create,
                [parent_label, lambda s: s.replace('_', ' '), Categorize()]
            ],
            train_setup=False,
            **kwargs
        )


# %% ../nbs/lfw.ipynb 6
from .pairs import *

# %% ../nbs/lfw.ipynb 7
class LFWPairsMixin(LFW):
    """Fixed pairs of facial images"""
    TEST_ITEMS_FILE_NAME = 'pairs.txt'
    DEV_TRAIN_ITEMS_FILE_NAME = 'pairsDevTrain.txt'
    DEV_TEST_ITEMS_FILE_NAME = 'pairsDevTest.txt'

    @return_list
    def _parse_items(self, fname):
        lines = self._fetch_file(fname).readlines()
        for l in lines[1:]:
            l = l.split()
            if len(l) == 3:
                name, num1, num2 = l
                yield [self._get_path(name, int(num1)),
                       self._get_path(name, int(num2))]
            else:
                name1, num1, name2, num2 = l
                yield [self._get_path(name1, int(num1)),
                       self._get_path(name2, int(num2))]
    
    def _load(self, **kwargs):
        return Datasets(
            tfms=[
                ImagePair.create,
                [lambda pair: parent_label(pair[0])==parent_label(pair[1]), Sameness()]
            ],
            train_setup=False,
            **kwargs
        )

# %% ../nbs/lfw.ipynb 8
class LFWPairs(LFWDevMixin, LFWPairsMixin, LFW):
    pass

# %% ../nbs/lfw.ipynb 11
class SLLFWPairs(LFWPairsMixin, LFW):
    """Similar Looking LFW: http://whdeng.cn/SLLFW/index.html"""
    BASE_URL = f'http://whdeng.cn/SLLFW'
    TEST_ITEMS_FILE_NAME = 'pair_SLLFW.txt'

    @return_list
    def _parse_items(self, fname):
        # Parsed according to http://www.whdeng.cn/SLLFW/index.html#download
        singles = re.findall(r'(.*)/.*_(\d*)', self._fetch_file(fname).read_text())
        pairs = [(singles[i], singles[i+1]) for i in range(0, len(singles), 2)]
        
        for ((name1, num1), (name2, num2)) in pairs:
            yield [self._get_path(name1, int(num1)),
                   self._get_path(name2, int(num2))]
