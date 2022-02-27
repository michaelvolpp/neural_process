import torch
from torch.utils.data import IterableDataset, Dataset
from metalearning_benchmarks.base_benchmark import MetaLearningBenchmark


class MetaLearningDataset(IterableDataset):
    def __init__(self, benchmark: MetaLearningBenchmark):
        super(MetaLearningDataset).__init__()
        self.benchmark = benchmark

    def __iter__(self):
        while True:
            yield self.benchmark.get_random_task()


class SingleTaskDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super(SingleTaskDataset).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]
