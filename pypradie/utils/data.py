# pypradie/utils/data.py

import numpy as np

class TensorDataset:
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.
    """

    def __init__(self, *tensors):
        """Initializes the dataset.

        Args:
            *tensors: Tensors that have the same size of the first dimension.
        """
        assert all(tensor.shape[0] == tensors[0].shape[0] for tensor in tensors), \
            "All tensors must have the same number of samples"
        self.tensors = tensors

    def __getitem__(self, index):
        """Gets a sample from the dataset.

        Args:
            index (int or slice): Index of the sample.

        Returns:
            tuple: A tuple of tensors corresponding to the sample.
        """
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        """Returns the total number of samples.

        Returns:
            int: Number of samples.
        """
        return self.tensors[0].shape[0]

class DataLoader:
    """Data loader that combines a dataset and a sampler, and provides an iterable over the dataset."""

    def __init__(self, dataset, batch_size=1, shuffle=False):
        """Initializes the data loader.

        Args:
            dataset (TensorDataset): The dataset to load data from.
            batch_size (int, optional): How many samples per batch to load. Defaults to 1.
            shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        """Returns the iterator object itself."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._current_batch = 0
        return self

    def __next__(self):
        """Returns the next batch of data.

        Returns:
            tuple: A batch of data from the dataset.

        Raises:
            StopIteration: If there are no more batches to return.
        """
        start_idx = self._current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        if start_idx >= len(self.dataset):
            raise StopIteration
        batch_indices = self.indices[start_idx:end_idx]
        self._current_batch += 1
        return self.dataset[batch_indices]

    def __len__(self):
        """Returns the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        total_samples = len(self.dataset)
        return (total_samples + self.batch_size - 1) // self.batch_size  # Ceiling division
