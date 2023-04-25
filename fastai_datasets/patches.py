# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Core/patches.ipynb.

# %% auto 0
__all__ = ['dl_defaults', 'ListToTuple']

# %% ../nbs/Core/patches.ipynb 4
import random
from collections import defaultdict
from typing import List, Dict, Sequence, Union
from functools import partial

from fastprogress.fastprogress import *
from fastai.vision.all import *

# %% ../nbs/Core/patches.ipynb 7
@patch
def sublist(self: TfmdLists, indices: Iterable[int]) -> TfmdLists:
    """a sublist that maintains laziness"""
    sub = self.new_empty()
    sub.items = [self.items[i] for i in indices]

    all_indices = L(range_of(self))
    def subsplit(s):
        split_idxs = set(all_indices[s])
        return [i for i, j in enumerate(indices) if j in split_idxs]
    sub.splits = [subsplit(s) for s in self.splits]
    
    return sub

# %% ../nbs/Core/patches.ipynb 11
@patch
def sub_dsets(self: Datasets, indices: Iterable[int]):
    return Datasets(tls=[t.sublist(indices) for t in self.tls])

# %% ../nbs/Core/patches.ipynb 15
@patch
def random_sub_dsets(self: Datasets, size, with_replacement=False, less_ok=False) -> Datasets:
    if size == 0:
        return self.subset([])
    if len(self) < size:
        assert less_ok
        size = len(self)
    sampler = random.choices if with_replacement else random.sample
    indices = sampler(range(len(self)),  k=size)
    return self.sub_dsets(indices)

# %% ../nbs/Core/patches.ipynb 20
@patch
def subset(self: TfmdLists, i):
    s = self._new(self._get(self.splits[i]), split_idx=i)
    s.splits = [slice(None), []]  # fastai bugfix
    return s

@patch
def __eq__(self: Union[Pipeline, Transform], other: Union[Pipeline, Transform]):
    """Needed to find shared transforms between TfmdLists"""
    return type(self) == type(other) and self.__dict__ == other.__dict__

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

# %% ../nbs/Core/patches.ipynb 27
@patch
def __add__(self: Datasets, other: Datasets):
    assert len(self.tls) == len(other.tls)
    return Datasets(tls=[t1 + t2 for t1, t2 in zip(self.tls, other.tls)])

# %% ../nbs/Core/patches.ipynb 32
@patch
def __sub__(self: Datasets, other: Datasets):
    assert self.tfms == other.tfms
    assert set(other.items).issubset(self.items)
    return self.sub_dsets([i for i, o in enumerate(self.items) if o not in set(other.items)])

# %% ../nbs/Core/patches.ipynb 35
@patch(as_prop=True)
def i2t(self: Datasets):
    assert self.n_inp == len(self.tls) - 1
    return self.tls[-1]

# %% ../nbs/Core/patches.ipynb 37
@patch(as_prop=True)
def by_target(self: Datasets) -> Dict[int, Datasets]:
    if not hasattr(self, '_by_target'):
        targets = [int(t) for t in progress_bar(self.i2t, comment='Class map: scanning targets')]
        class_map = groupby(enumerate(targets), key=1, val=0)
        self._by_target = {self.vocab[c]: self.sub_dsets(indices)
                           for c, indices in progress_bar(class_map.items(), comment='Class map: partitioning')}
    return self._by_target


# %% ../nbs/Core/patches.ipynb 39
import matplotlib.pyplot as plt

@patch()
def plot_class_distribution(self: Datasets):
    for split in self.subsets:
        plt.bar(self.vocab, [len(split.by_target[c]) for c in self.vocab])

# %% ../nbs/Core/patches.ipynb 43
class ListToTuple(Transform):
    """Transforms lists to tuples, useful for fixing a bug in pytorch (pin_memory turns inner tuples into lists)"""
    def encodes(self, o:list):
        return tuple(o)


# %% ../nbs/Core/patches.ipynb 44
dl_defaults = {'pin_memory': default_device() != torch.device('cpu'), 'device': default_device(),
               'after_item': [ToTensor], 'after_batch': [ListToTuple, IntToFloatTensor]}

# %% ../nbs/Core/patches.ipynb 46
def _dl_args(kwargs):
    args = deepcopy(dl_defaults)
    for event in ['after_item', 'after_batch']:
        if event in kwargs:
            tfms = kwargs[event]
            args[event] += tfms if isinstance(tfms, Sequence) else [tfms]
    return args


@patch
def dls(self: Datasets, **kwargs) -> DataLoaders:
    """Calls `Datasets.dataloaders` with defaults from `dl_defaults`"""
    return self.dataloaders(**_dl_args(kwargs))


@patch
def dl(self: Datasets, **kwargs) -> DataLoader:
    """Creates a `DataLoader` (ignoring splits) with defaults from `dl_defaults`"""
    return self._dl_type(self, **_dl_args(kwargs))

# %% ../nbs/Core/patches.ipynb 48
@patch
def load(self: Datasets, **kwargs):
    return first(self.dl(bs=len(self), **kwargs))

# %% ../nbs/Core/patches.ipynb 51
@patch(as_prop=True)
def subsets(self: Datasets) -> TfmdLists:
    """Lazy list of a `Datasets`'s subsets"""
    return TfmdLists(range(self.n_subsets), self.subset)

# %% ../nbs/Core/patches.ipynb 53
@patch
def resplit(self: Datasets,
            splits: Union[Callable, List[List[int]]]  # a splitter function or a list of splits
            ):
    """Sets the splits of a `Datasets`"""
    if isinstance(splits, Callable):
        splits = splits(self)
    for t in self.tls:
        t.splits = splits

# %% ../nbs/Core/patches.ipynb 56
@patch()
def __repr__(self: Datasets):
    return '['+'\n'.join(repr(s) for s in self.subsets)+']' if self.split_idx is None else coll_repr(self)
