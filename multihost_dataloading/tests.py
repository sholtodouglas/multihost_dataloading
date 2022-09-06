from email.policy import default
import jax
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
import numpy as np
from collections import defaultdict

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)

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

def device_to_host(devices):
  '''Gets a mapping from device ID to host ID.
  
  Args:
    devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0),..]
  Returns:
    mapping: {device_id: host_id,..}
    host_devices: {host_id: [device_1, device_2],..}
  '''
  mapping = {}
  host_devices = defaultdict(list)
  for d in devices:
    mapping[d.id] = d.host_id
    host_devices[d.host_id].append(d.id) # default dict so no need to check

  return mapping, host_devices


# TODO: Check this function and dehackify it 
def construct_test_mesh_32(host_devices):
  '''
  By default, when we reshape a 32 (i.e. 4 hosts, 8 devices each) slice to (data, model) 
  unless the length of the model dimension is great than the number of devices per host,
  it will not be arranged with the second dimension crossing host boundaries. 

  This makes sense - but we want to test the most general case: where a given host may have
  multiple independent model replicas on it (each loading different data), and where these 
  replicas may stretch across host boundaries. 

  This may occur! E.g. PaLM used 12 way model parallelism in a given replica.
  Therefore the layout may have been that one host had 8, the next 4 + 4, the next 8?

  To test this, we want a layout that looks like this, indices indicate host idx of devices

      00001111
      00001111
      22223333
      22223333
  
  Args:
    host_devices: {host_id: [device_1, device_2],..}

  '''
  assert sum([len(devices) for _, devices in host_devices.items()]) == 32

  test_mesh_ids = [
    host_devices[0][0:4] + host_devices[1][0:4],
    host_devices[0][4:8] + host_devices[1][4:8],
    host_devices[2][0:4] + host_devices[3][0:4],
    host_devices[2][4:8] + host_devices[3][4:8],
  ]

  return test_mesh_ids



def test_one():
  '''
  
  Args: 
    data_dim: How many devices to put along the data dimension
  '''
  devices = jax.devices()
  mapping, host_devices = device_to_host(devices)
  test_mesh_layout = construct_test_mesh_32(host_devices)

  print(test_mesh_layout)


test_one()


  # global_mesh = Mesh(np.array(devices).reshape(data_dim, len(devices)//data_dim), ('x', 'y'))