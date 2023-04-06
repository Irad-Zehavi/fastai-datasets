from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Dict, Sequence, Union
from tqdm.auto import tqdm

from fastai.vision.all import *


class ToTuple(Transform):
    # bugfix (pin_memory for inner tupples)
    def encodes(self, o:list):
        return tuple(o) 
    

_dl_defaults = {'pin_memory': True, 'bs': 2 ** 8, 'device': default_device(),
                'after_item': [ToTensor], 'after_batch': [ToTuple, IntToFloatTensor]}


@patch
def __eq__(self: Union[Pipeline, Transform], other: Union[Pipeline, Transform]):
    return type(self) == type(other) and self.__dict__ == other.__dict__


# TfmdLists

@patch
def __add__(l1: TfmdLists, l2: TfmdLists):
    assert l1.split_idx == l2.split_idx

    tfms1, tfms2 = copy(list(l1.tfms)), copy(list(l2.tfms))
    merged_tfms = []
    while tfms1 and tfms2 and tfms1[-1] == tfms2[-1]:
        merged_tfms.insert(0, tfms1.pop())
        tfms2.pop()
    tfms1, tfms2 = Pipeline(tfms1), Pipeline(tfms2)

    return TfmdLists(
        [[i, item] for i, l in enumerate([l1, l2]) for item in l.items],
        tfms=[lambda o: [tfms1, tfms2][o[0]](o[1]), *merged_tfms],
        splits=[L(range_of(l1))[s1] + [i+len(l1) for i in L(range_of(l2))[s2]]
                for s1, s2 in zip_longest(l1.splits, l2.splits, fillvalue=[])],
        do_setup=False
    )


@patch
def subset(self: TfmdLists, i):
    s = self._new(self._get(self.splits[i]), split_idx=i)
    s.splits = [slice(None), []]
    return s


@patch
def sublist(self: TfmdLists, indices: Iterable[int]):
    sub = self.new_empty()
    sub.items = [self.items[i] for i in indices]

    all_indices = L(range_of(self))
    def subsplit(s):
        split_idxs = all_indices[s]
        return [i for i, j in enumerate(indices) if j in split_idxs]
    sub.splits = [subsplit(s) for s in self.splits]
    
    return sub


# Datasets

@patch
def __add__(self: Datasets, other: Datasets):
    assert len(self.tls) == len(other.tls)
    return Datasets(tls=[t1 + t2 for t1, t2 in zip(self.tls, other.tls)])


@patch
def sub_dsets(self: Datasets, indices: Iterable[int]):
    return Datasets(tls=[t.sublist(indices) for t in self.tls])


@patch
def __sub__(self: Datasets, other: Datasets):
    assert self.tfms == other.tfms
    assert not (set(self.items) - set(other.tfms))
    return self.sub_dsets(self, [i for i, o in enumerate(self.items) if o not in other.items])


@patch
def random_sub_dsets(self: Datasets, size, with_replacement=False, less_ok=False) -> Datasets:
    if size == 0:
        return self.subset([])
    if len(self) < size and not with_replacement and less_ok:
        size = len(self)
    if with_replacement:
        indices = random.choices(range(len(self)), k=size)
    else:
        indices = random.sample(range(len(self)), size)
    return self.sub_dsets(indices)


@patch(as_prop=True)
def targets(self: Datasets):
    assert self.n_inp == len(self.tls) - 1
    return self.tls[-1]


@patch(as_prop=True)
def by_class(self: Datasets) -> Dict[int, Datasets]:
    if not hasattr(self, '_by_class'):
        targets = [self.vocab[t] for t in tqdm(self.targets, desc='Class map: scanning targets')]
        class_map = groupby(enumerate(targets), 1, 0)
        self._by_class = {c: self.sub_dsets(indices) for c, indices in tqdm(class_map.items(), desc='Class map: partitioning')}
    return self._by_class


@patch
def resplit(self: Datasets, splits: Union[Callable, List[List[int]]]):
    if isinstance(splits, Callable):
        splits = splits(self)
    for t in self.tls:
        t.splits = splits


def _dl_args(kwargs):
    args = deepcopy(_dl_defaults)
    for event in ['after_item', 'after_batch']:
        if event in kwargs:
            tfms = kwargs[event]
            args[event] += tfms if isinstance(tfms, Sequence) else [tfms]
    return args


@patch
def dls(self: Datasets, **kwargs):
    return self.dataloaders(**_dl_args(kwargs))


@patch
def dl(self: Datasets, **kwargs):
    return self._dl_type(self, **_dl_args(kwargs))


@patch
def load(self: Datasets, **kwargs):
    return first(self.dl(bs=len(self), **kwargs))


@patch
def subsets(self: Datasets):
    return [self.subset(i) for i in range(self.n_subsets)]
