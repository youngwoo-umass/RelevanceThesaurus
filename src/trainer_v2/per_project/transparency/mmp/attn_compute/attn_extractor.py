import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from misc_lib import batch_iter_from_entry_iter


# TODO
#   Iterate triplet
#   Format as


def work(triplet: Iterable[Tuple[str, str, str]], extract):
    # query / document / score / attention
    Item = Tuple[str, str, float, np.array]
    all_items: List[Tuple[Item, Item]] = []
    for batch in batch_iter_from_entry_iter(triplet, 1000):
        items = extract(batch)
        all_items.extend(items)



