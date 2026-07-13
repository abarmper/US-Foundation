"""Task-homogeneous batch samplers (verbatim from the original samplers_final.py).

Every batch is a single task, which the model's per-batch head routing requires.
"""

import random
from collections import defaultdict
from torch.utils.data import Sampler


class HomogeneousTaskSampler(Sampler):
    """Batches where every image belongs to the same task_id (validation)."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.task_indices = defaultdict(list)
        for idx in range(len(dataset)):
            self.task_indices[dataset.samples[idx]["task_id"]].append(idx)

        self.batches = []
        for task_id, indices in self.task_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) > 0:   # keep tiny tasks (IVC/PSAX) from vanishing
                    self.batches.append(batch)
        random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class DeterministicBalancedHomogeneousSampler(Sampler):
    """Every task sees the same number of batches each epoch (training)."""

    def __init__(self, task_labels, batch_size, num_batches_per_epoch):
        self.task_labels = task_labels
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.task_indices = {}
        for idx, task in enumerate(self.task_labels):
            self.task_indices.setdefault(task, []).append(idx)

        self.available_tasks = sorted(self.task_indices.keys())
        self.num_tasks = len(self.available_tasks)

    def __iter__(self):
        shuffled = {}
        for task in self.available_tasks:
            shuffled[task] = self.task_indices[task].copy()
            random.shuffle(shuffled[task])

        task_cycle_idx = 0
        for _ in range(self.num_batches_per_epoch):
            current_task = self.available_tasks[task_cycle_idx]
            batch = []
            for _ in range(self.batch_size):
                if not shuffled[current_task]:
                    shuffled[current_task] = self.task_indices[current_task].copy()
                    random.shuffle(shuffled[current_task])
                batch.append(shuffled[current_task].pop())
            yield batch
            task_cycle_idx = (task_cycle_idx + 1) % self.num_tasks

    def __len__(self):
        return self.num_batches_per_epoch


class TemperatureTaskSampler(Sampler):
    """Task-homogeneous batches with temperature-controlled task frequency.

    Each task gets a number of batches per epoch proportional to n_task**temperature
    (n_task = its sample count), so:
      * temperature=0 -> every task equal (like the balanced sampler),
      * temperature=1 -> natural frequency,
      * 0<temperature<1 (e.g. 0.5, sqrt) -> a middle ground that stops the tiny
        cardiac tasks from being replayed ~100x/epoch and overfitting.
    Every task keeps at least one batch. Total batches ~= num_batches_per_epoch.
    """

    def __init__(self, task_labels, batch_size, num_batches_per_epoch, temperature):
        self.batch_size = batch_size
        self.task_indices = {}
        for idx, task in enumerate(task_labels):
            self.task_indices.setdefault(task, []).append(idx)
        self.tasks = sorted(self.task_indices)

        weights = {t: len(self.task_indices[t]) ** temperature for t in self.tasks}
        tot = sum(weights.values())
        counts = {t: max(1, round(num_batches_per_epoch * weights[t] / tot)) for t in self.tasks}
        self.schedule = []
        for t in self.tasks:
            self.schedule += [t] * counts[t]

    def __iter__(self):
        pools = {t: self.task_indices[t].copy() for t in self.tasks}
        for lst in pools.values():
            random.shuffle(lst)
        order = self.schedule.copy()
        random.shuffle(order)
        for task in order:
            batch = []
            for _ in range(self.batch_size):
                if not pools[task]:
                    pools[task] = self.task_indices[task].copy()
                    random.shuffle(pools[task])
                batch.append(pools[task].pop())
            yield batch

    def __len__(self):
        return len(self.schedule)
