import bisect


class Dataset(object):
    """
    An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


if __name__ == "__main__":
    class CDataset(Dataset):
        def __init__(self):
            pass

        def __getitem__(self, index):
            return {"i": index}

        def __len__(self):
            return 10


    from torocr.dataflow.dataloader_v2 import DataLoader
    from torocr.dataflow.sampler import DistributedSampler

    d = CDataset()
    # sampler = DistributedSampler(d, shuffle=True,
    #                              num_replicas=2)

    loader = DataLoader(dataset=d, num_workers=4, batch_size=2)
    # print(len(loader))
    # import time
    # time.sleep(3)
    # for i in range(3):
    #     for j, batch in enumerate(loader):
    #         print(batch, i, j)

    # ii = loader.__iter__()
    # for i in range(6):
    #     a = next(ii)
    #     print(a)

    for i, batch in enumerate(loader):
        print(i, batch, 888)

    for i, batch in enumerate(loader):
        print(i, batch, 888)

    # loader = DataLoader(dataset=d, batch_size=2, num_workers=1)
    # for i, batch in enumerate(loader):
    #     print(i, batch)
    # import numpy as np
    # np.random.seed(0)
    # print(np.random.rand(4, 3))
    #
    # # np.random.seed(0)
    # print(np.random.rand(4, 3))
