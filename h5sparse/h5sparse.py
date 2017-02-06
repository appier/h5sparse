import h5py
import numpy as np
import scipy.sparse as ss


class Group(object):
    """
    Parameters
    ==========
    h5py_group: h5py.Group
    """

    def __init__(self, h5py_group):
        self.h5py_group = h5py_group

    def __getitem__(self, key):
        h5py_item = self.h5py_group[key]
        if isinstance(h5py_item, h5py.Group):
            if set(h5py_item.keys()) == set(['data', 'indices', 'indptr',
                                             'shape']):
                # detect the sparse matrix
                return Dataset(h5py_item)
            else:
                return Group(h5py_item)
        elif isinstance(h5py_item, h5py.Dataset):
            return h5py_item
        else:
            raise ValueError("Unexpected item type.")

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       **kwargs): # pylint: disable=unused-argument
        """Create 4 datasets in a group to represent the sparse array"""
        if data is None:
            raise NotImplementedError("Only support create_dataset with "
                                      "existed data.")
        elif isinstance(data, Dataset):
            self.h5py_group.create_dataset(
                name + '/data', data=data.h5py_group['data'], **kwargs)
            self.h5py_group.create_dataset(
                name + '/indices', data=data.h5py_group['indices'], **kwargs)
            self.h5py_group.create_dataset(
                name + '/indptr', data=data.h5py_group['indptr'], **kwargs)
            self.h5py_group.create_dataset(
                name + '/shape', data=data.h5py_group['shape'])
        else:
            self.h5py_group.create_dataset(
                name + '/data', shape=data.data.shape, data=data.data,
                **kwargs)
            self.h5py_group.create_dataset(
                name + '/indices', shape=data.indices.shape, data=data.indices,
                **kwargs)
            self.h5py_group.create_dataset(
                name + '/indptr', shape=data.indptr.shape, data=data.indptr,
                **kwargs)
            self.h5py_group.create_dataset(
                name + '/shape', data=np.asarray(data.shape))


class File(Group):
    """
    Parameters
    ==========
    *args, **kwargs: the parameters from h5py.File
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        self.h5f = h5py.File(*args, **kwargs)
        self.h5py_group = self.h5f

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5f.__exit__(exc_type, exc_value, traceback)


class Dataset(object):
    """
    Parameters
    ==========
    h5py_group: h5py.Dataset
    """

    def __init__(self, h5py_group):
        self.h5py_group = h5py_group

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("Index step is not supported.")
            start = key.start
            stop = key.stop
            if stop is not None and stop > 0:
                stop += 1
            if start is not None and start < 0:
                start -= 1
            indptr_slice = slice(start, stop)
            indptr = self.h5py_group['indptr'][indptr_slice]
            data = self.h5py_group['data'][indptr[0]:indptr[-1]]
            indices = self.h5py_group['indices'][indptr[0]:indptr[-1]]
            indptr -= indptr[0]
            shape = (indptr.size - 1, self.h5py_group['shape'][1])
        else:
            raise NotImplementedError("Only support one slice as index.")

        return ss.csr_matrix((data, indices, indptr), shape=shape)

    @property
    def value(self):
        data = self.h5py_group['data'].value
        indices = self.h5py_group['indices'].value
        indptr = self.h5py_group['indptr'].value
        shape = self.h5py_group['shape'].value
        return ss.csr_matrix((data, indices, indptr), shape=shape)
