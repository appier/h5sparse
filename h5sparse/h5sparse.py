import six
import h5py
import scipy.sparse as ss


FORMAT_DICT = {
    'csr': ss.csr_matrix,
    'csc': ss.csc_matrix,
}


def get_format_str(data):
    for format_str, format_class in six.viewitems(FORMAT_DICT):
        if isinstance(data, format_class):
            return format_str
    raise ValueError("Data type {} is not supported.".format(type(data)))


def get_format_class(format_str):
    format_class = FORMAT_DICT.get(format_str, None)
    if format_class is None:
        raise ValueError("Format string {} is not supported."
                         .format(format_str))
    return format_class


class Group(object):
    """The HDF5 group that can detect and create sparse matrix.

    Parameters
    ==========
    h5py_group: h5py.Group
    """

    def __init__(self, h5py_group):
        self.h5py_group = h5py_group

    def __getitem__(self, key):
        h5py_item = self.h5py_group[key]
        if isinstance(h5py_item, h5py.Group):
            if 'h5sparse_format' in h5py_item.attrs:
                # detect the sparse matrix
                return Dataset(h5py_item)
            else:
                return Group(h5py_item)
        elif isinstance(h5py_item, h5py.Dataset):
            return h5py_item
        else:
            raise ValueError("Unexpected item type.")

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       format='csr', **kwargs):
        """Create 4 datasets in a group to represent the sparse array."""
        if data is None:
            raise NotImplementedError("Only support create_dataset with "
                                      "existed data.")
        elif isinstance(data, Dataset):
            group = self.h5py_group.create_group(name)
            group.attrs['h5sparse_format'] = data.h5py_group.attrs['h5sparse_format']
            group.attrs['h5sparse_shape'] = data.h5py_group.attrs['h5sparse_shape']
            group.create_dataset('data', data=data.h5py_group['data'], **kwargs)
            group.create_dataset('indices', data=data.h5py_group['indices'], **kwargs)
            group.create_dataset('indptr', data=data.h5py_group['indptr'], **kwargs)
        else:
            group = self.h5py_group.create_group(name)
            group.attrs['h5sparse_format'] = get_format_str(data)
            group.attrs['h5sparse_shape'] = data.shape
            group.create_dataset('data', data=data.data, **kwargs)
            group.create_dataset('indices', data=data.indices, **kwargs)
            group.create_dataset('indptr', data=data.indptr, **kwargs)


class File(Group):
    """The HDF5 file object that can detect and create sparse matrix.

    Parameters
    ==========
    *args, **kwargs: the parameters from h5py.File
    """

    def __init__(self, *args, **kwargs):
        self.h5f = h5py.File(*args, **kwargs)
        self.h5py_group = self.h5f

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5f.__exit__(exc_type, exc_value, traceback)


class Dataset(object):
    """The HDF5 sparse matrix dataset.

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
            shape = (indptr.size - 1,
                     self.h5py_group.attrs['h5sparse_shape'][1])
        else:
            raise NotImplementedError("Only support one slice as index.")

        format_class = get_format_class(self.h5py_group.attrs['h5sparse_format'])
        return format_class((data, indices, indptr), shape=shape)

    @property
    def value(self):
        data = self.h5py_group['data'].value
        indices = self.h5py_group['indices'].value
        indptr = self.h5py_group['indptr'].value
        shape = self.h5py_group.attrs['h5sparse_shape']
        format_class = get_format_class(self.h5py_group.attrs['h5sparse_format'])
        return format_class((data, indices, indptr), shape=shape)
