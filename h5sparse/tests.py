import os
import json
from tempfile import mkstemp

import numpy as np
import scipy.sparse as ss

import h5sparse


def test_create_and_read_dataset():
    h5_path = mkstemp(suffix=".h5")[1]
    sparse_matrix = ss.csr_matrix([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0],
                                   [1, 1, 0]],
                                  dtype=np.float64)
    with h5sparse.File(h5_path) as h5f:
        h5f.create_dataset('sparse/matrix', data=sparse_matrix)
    with h5sparse.File(h5_path) as h5f:
        assert 'sparse' in h5f
        assert 'matrix' in h5f['sparse']
        assert (h5f['sparse']['matrix'][1:3] != sparse_matrix[1:3]).size == 0
        assert (h5f['sparse']['matrix'][2:] != sparse_matrix[2:]).size == 0
        assert (h5f['sparse']['matrix'][:2] != sparse_matrix[:2]).size == 0
        assert (h5f['sparse']['matrix'][-2:] != sparse_matrix[-2:]).size == 0
        assert (h5f['sparse']['matrix'][:-2] != sparse_matrix[:-2]).size == 0
        assert (h5f['sparse']['matrix'].value != sparse_matrix).size == 0

    os.remove(h5_path)


def test_create_dataset_from_dataset():
    from_h5_path = mkstemp(suffix=".h5")[1]
    to_h5_path = mkstemp(suffix=".h5")[1]
    sparse_matrix = ss.csr_matrix([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0],
                                   [1, 1, 0]],
                                  dtype=np.float64)
    with h5sparse.File(from_h5_path) as from_h5f:
        from_dset = from_h5f.create_dataset('sparse/matrix', data=sparse_matrix)

        with h5sparse.File(to_h5_path) as to_h5f:
            to_h5f.create_dataset('sparse/matrix', data=from_dset)
            assert 'sparse' in to_h5f
            assert 'matrix' in to_h5f['sparse']
            assert (to_h5f['sparse/matrix'].value != sparse_matrix).size == 0

    os.remove(from_h5_path)
    os.remove(to_h5_path)


def test_dataset_append():
    h5_path = mkstemp(suffix=".h5")[1]
    sparse_matrix = ss.csr_matrix([[0, 1, 0],
                                   [0, 0, 1],
                                   [0, 0, 0],
                                   [1, 1, 0]],
                                  dtype=np.float64)
    to_append = ss.csr_matrix([[0, 1, 1],
                               [1, 0, 0]],
                              dtype=np.float64)
    appended_matrix = ss.vstack((sparse_matrix, to_append))

    with h5sparse.File(h5_path) as h5f:
        h5f.create_dataset('matrix', data=sparse_matrix, chunks=(100000,),
                           maxshape=(None,))
        h5f['matrix'].append(to_append)
        assert (h5f['matrix'].value != appended_matrix).size == 0

    os.remove(h5_path)


def test_numpy_array():
    h5_path = mkstemp(suffix=".h5")[1]
    matrix = np.random.rand(3, 5)
    with h5sparse.File(h5_path) as h5f:
        h5f.create_dataset('matrix', data=matrix)
        assert 'matrix' in h5f
        np.testing.assert_equal(h5f['matrix'].value, matrix)
    os.remove(h5_path)


def test_bytestring():
    h5_path = mkstemp(suffix=".h5")[1]
    strings = [str(i) for i in range(100)]
    data = json.dumps(strings).encode('utf8')
    with h5sparse.File(h5_path) as h5f:
        h5f.create_dataset('strings', data=data)
        assert 'strings' in h5f
        assert strings == json.loads(h5f['strings'].value.decode('utf8'))
    os.remove(h5_path)
