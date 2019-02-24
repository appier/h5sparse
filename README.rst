h5sparse
========
.. image:: https://img.shields.io/travis/appier/h5sparse/master.svg
   :target: https://travis-ci.org/appier/h5sparse
.. image:: https://img.shields.io/pypi/v/h5sparse.svg
   :target: https://pypi.python.org/pypi/h5sparse
.. image:: https://img.shields.io/pypi/l/h5sparse.svg
   :target: https://pypi.python.org/pypi/h5sparse

Scipy sparse matrix in HDF5.


Installation
------------
.. code:: bash

   pip install h5sparse


Testing
-------
- for single environment:

  .. code:: bash

     python setup.py test

- for all environments:

  .. code:: bash

     tox


Examples
--------

Create dataset
**************
.. code:: python

   In [1]: import scipy.sparse as ss
      ...: import h5sparse
      ...: import numpy as np
      ...:

   In [2]: sparse_matrix = ss.csr_matrix([[0, 1, 0],
      ...:                                [0, 0, 1],
      ...:                                [0, 0, 0],
      ...:                                [1, 1, 0]],
      ...:                               dtype=np.float64)

   In [3]: # create dataset from scipy sparse matrix
      ...: with h5sparse.File("test.h5") as h5f:
      ...:     h5f.create_dataset('sparse/matrix', data=sparse_matrix)

   In [4]: # you can also create dataset from another dataset
      ...: with h5sparse.File("test.h5") as h5f:
      ...:     h5f.create_dataset('sparse/matrix2', data=h5f['sparse/matrix'])

   In [5]: # you can also create dataset using the formats that original h5py accepts
      ...: with h5sparse.File("test.h5") as h5f:
      ...:     h5f.create_dataset('sparse/matrix3', data=[1,2,3])

Read dataset
************
.. code:: python

   In [6]: h5f = h5sparse.File("test.h5")

   In [7]: h5f['sparse/matrix'][1:3]
   Out[7]:
   <2x3 sparse matrix of type '<class 'numpy.float64'>'
           with 1 stored elements in Compressed Sparse Row format>

   In [8]: h5f['sparse/matrix'][1:3].toarray()
   Out[8]:
   array([[ 0.,  0.,  1.],
          [ 0.,  0.,  0.]])

   In [9]: h5f['sparse']['matrix'][1:3].toarray()
   Out[9]:
   array([[ 0.,  0.,  1.],
          [ 0.,  0.,  0.]])

   In [10]: h5f['sparse']['matrix'][2:].toarray()
   Out[10]:
   array([[ 0.,  0.,  0.],
          [ 1.,  1.,  0.]])

   In [11]: h5f['sparse']['matrix'][:2].toarray()
   Out[11]:
   array([[ 0.,  1.,  0.],
          [ 0.,  0.,  1.]])

   In [12]: h5f['sparse']['matrix'][-2:].toarray()
   Out[12]:
   array([[ 0.,  0.,  0.],
          [ 1.,  1.,  0.]])

   In [13]: h5f['sparse']['matrix'][:-2].toarray()
   Out[13]:
   array([[ 0.,  1.,  0.],
          [ 0.,  0.,  1.]])

   In [14]: h5f['sparse']['matrix'][()].toarray()
   Out[14]:
   array([[ 0.,  1.,  0.],
          [ 0.,  0.,  1.],
          [ 0.,  0.,  0.],
          [ 1.,  1.,  0.]])

   In [15]: import h5py

   In [16]: h5py_h5f = h5py.File("test.h5")

   In [17]: h5sparse.Group(h5py_h5f.id)['sparse/matrix'][()]
   Out[17]:
   <4x3 sparse matrix of type '<class 'numpy.float64'>'
           with 4 stored elements in Compressed Sparse Row format>

   In [18]: h5sparse.Group(h5py_h5f['sparse'].id)['matrix'][()]
   Out[18]:
   <4x3 sparse matrix of type '<class 'numpy.float64'>'
           with 4 stored elements in Compressed Sparse Row format>

   In [19]: h5sparse.Dataset(h5py_h5f['sparse/matrix'])[()]
   Out[19]:
   <4x3 sparse matrix of type '<class 'numpy.float64'>'
           with 4 stored elements in Compressed Sparse Row format>

Append dataset
**************
.. code:: python

   In [20]: to_append = ss.csr_matrix([[0, 1, 1],
       ...:                            [1, 0, 0]],
       ...:                           dtype=np.float64)

   In [21]: h5f.create_dataset('matrix', data=sparse_matrix, chunks=(100000,),
       ...:                    maxshape=(None,))

   In [22]: h5f['matrix'].append(to_append)

   In [23]: h5f['matrix'][()]
   Out[23]:
   <6x3 sparse matrix of type '<class 'numpy.float64'>'
           with 7 stored elements in Compressed Sparse Row format>

   In [24]: h5f['matrix'][()].toarray()
   Out[24]:
   array([[ 0.,  1.,  0.],
          [ 0.,  0.,  1.],
          [ 0.,  0.,  0.],
          [ 1.,  1.,  0.],
          [ 0.,  1.,  1.],
          [ 1.,  0.,  0.]])
