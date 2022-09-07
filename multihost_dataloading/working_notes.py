
'''
Notes explaining how to think about GDA partitioning. 

Imagine that you have a single 8 device TPU, which we lay out (4,2)

>>> import jax
>>> from jax.experimental.maps import Mesh
>>> from jax.experimental import PartitionSpec as P
>>> from jax.experimental.global_device_array import GlobalDeviceArray
>>> import numpy as np
>>> global_mesh = global_mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))

Create an 8x8 input to demonstrate partitioning.

>>> global_input_shape = (8, 8)
>>> global_input_data = np.arange(np.prod(global_input_shape)).reshape(global_input_shape)
array([
       [ 0,  1,  2,  3, | 4,  5,  6,  7],
       [ 8,  9, 10, 11, | 12, 13, 14, 15],
       [16, 17, 18, 19, | 20, 21, 22, 23],
       [24, 25, 26, 27, | 28, 29, 30, 31],
       [32, 33, 34, 35, | 36, 37, 38, 39],
       [40, 41, 42, 43, | 44, 45, 46, 47],
       [48, 49, 50, 51, | 52, 53, 54, 55],
       [56, 57, 58, 59, | 60, 61, 62, 63]
                                        ])

To construct a GlobalDeviceArray, we must construct a function which takes index
as input. This index is an up to 3 dimensional slice of the global input shape which
depends 'mesh_axes' and the individual device position in the global mesh, indicating
how the GDA is partitioned across devices. As an axis, None means replicated. 

E.g.

>>> def cb(index):
...  return global_input_data[index] 

>>> gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh, P('x', None), cb)
>>> gda.local_data(0)

index for device 0:

(slice(0, 4, None), slice(None, None, None))

DeviceArray([[ 0,  1,  2,  3],
             [ 8,  9, 10, 11],
             [16, 17, 18, 19],
             [24, 25, 26, 27],
             [32, 33, 34, 35],
             [40, 41, 42, 43],
             [48, 49, 50, 51],
             [56, 57, 58, 59]], dtype=int32)


>>> gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh, P(None, 'y'), cb)
>>> gda.local_data(0)

(slice(None, None, None), slice(0, 2, None))

DeviceArray([[ 0,  1],
             [ 8,  9],
             [16, 17],
             [24, 25],
             [32, 33],
             [40, 41],
             [48, 49],
             [56, 57]], dtype=int32)

>>> gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh, P('x', 'y'), cb)
>>> gda.local_data(0)

(slice(0, 4, None), slice(0, 2, None))
DeviceArray([[ 0,  1],
             [ 8,  9],
             [16, 17],
             [24, 25]], dtype=int32)
'''