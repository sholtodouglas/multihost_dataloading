from email.policy import default
import jax
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental import global_device_array as gda_lib
import numpy as np
from collections import defaultdict

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)


def device_to_host(devices):
  '''Gets mapping from/to device ID <> host ID.
  
  Use the host->devices mapping to specify global device layouts that all hosts
  have visibility over (otherwise they would only have the indices of theirs).


  Args:
    devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0),..]
  Returns:
    device_host_map: {device_id: host_id,..}
    host_devices_map: {host_id: [device_1, device_2],..}
  '''

  host_devices_map = defaultdict(list)
  for d in devices:
    host_devices_map[d.host_id].append(d) # default dict so no need to check

  return host_devices_map


# TODO: Check this function and dehackify it 
def construct_test_mesh_32(host_devices_map):
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
  

  assert sum([len(devices) for _, devices in host_devices_map.items()]) == 32

  test_mesh_layout = np.array([
    host_devices_map[0][0:4] + host_devices_map[1][0:4],
    host_devices_map[0][4:8] + host_devices_map[1][4:8],
    host_devices_map[2][0:4] + host_devices_map[3][0:4],
    host_devices_map[2][4:8] + host_devices_map[3][4:8],
  ])
  assert test_mesh_layout.shape == (4, 8)

  return test_mesh_layout

def deduplicate_slices(local_slices):
  '''
  [(slice(4, 6, None), slice(None, None, None)), (slice(4, 6, None), slice(None, None, None)), (slice(6, 8, None), slice(None, None, None)), (slice(6, 8│one, None, None)), (slice(2, 4, None), slice(None, None, None)), (slice(2, 4, None), slice(None, None, Non
  , None), ...]
  
  Returns the unique set of slices we need to load locally
  And the mapping of device to those slices
  '''
  unique_slices = {}
  device_to_slice = {}

  for (device, slice_tuple) in local_slices:
    slice_hash = gda_lib._hashed_index(slice_tuple)
    if slice_hash not in unique_slices:
      unique_slices[slice_hash] = slice_tuple # d
    device_to_slice[device] = slice_hash

  # unique_slices : {((4, 6), (None, None)): (slice(4, 6, None), slice(None, None, None)), ((6, 8), (None, None)): (slice(6, 8, None), slice(None, None, None))}
  # device_to_slice: [{TpuDevice(id=16, process_index=2, coords=(0,2,0), core_on_chip=0): ((4, 6), (None, None)),...]
  return unique_slices, device_to_slice

def load_slices_to_host(global_data, pipelines):
  ''' For the moment just load from an array. TODO: Load from tf.data
  The device to pipeline hash will share the same hash here, so we
  can directly put from these buffers into local_device_buffers to
  form a GDA.
  '''
  unique_host_buffers = {}
  for hash, slice in pipelines.items():
    unique_host_buffers[hash] = global_data[slice]
  return unique_host_buffers

def construct_GDA(global_data_shape, device_mesh, data_axes, device_to_pipeline, unique_host_buffers):

  device_buffers = []
  for device, pipeline_hash in device_to_pipeline.items():
    host_data = unique_host_buffers[pipeline_hash]
    device_buffers.append(jax.device_put(host_data, device))

  return GlobalDeviceArray(global_data_shape, device_mesh, data_axes, device_buffers)

  # def _gda(global_data_shape, dbs):
  #   return GlobalDeviceArray(global_data_shape, device_mesh, data_axes, dbs)
  # return jax.tree_map(
  #     _gda,
  #     device_buffers,
  #     global_data_shape)
  


def test_one():
  '''
  We follow these steps:

  1. Initialise our desired GDA shape and device mesh layout
  2. Get the slices of the GDA corresponding to each device
  3. For each host, identify which slices it needs to load to feed it's devices
  4. Deduplicate these slices
  5. Load the slices (in this test, from an array - TODO: from .tfrecords)
  6. Load them into the local device buffers and wrap it as one big GDA


  '''

  # 1. Initialise our desired GDA shape and deivce mesh layout
  # Construct global data
  global_data_shape = (8, 4)
  global_data = np.arange(np.prod(global_data_shape)).reshape(global_data_shape)
  data_axes = P('data', None)
  # Create our device mesh - this function arranges a 4 hosts/32 devices
  # to allow us to test the general case
  devices = jax.devices()
  host_devices_map = device_to_host(devices)
  test_mesh_layout = construct_test_mesh_32(host_devices_map)
  # imagine these integers are host id, and the numbers are each a device
  # we create four replicas, each split over two hosts
  #     00001111
  #     00001111
  #     22223333
  #     22223333
  global_mesh = Mesh(test_mesh_layout, ('data', 'model'))

  # 2. Get the slices of the GDA corresponding to each device
  # returns e.g. [TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1):
  #                                   (slice(6, 8, None), slice(None, None, None)),]
  devices_to_slices = gda_lib.get_shard_indices(global_data_shape, global_mesh, data_axes)

  all_slices = [(device, devices_to_slices[device]) for device in devices]

  # What the t5x example does here is it splits the total length of the input data array
  #  by num unique slices and assigns each slice (or 'pipeline') to a host.
  # This doesn't account for the situation where multiple hosts might WANT to load the
  # same data. E.g in our case there are 4 unique slices ('pipelines'), but what we want
  # to test is each host loading the data required of it and not needing to reshard
  # during pjit. Instead, what we will do is:

  # 
  # 3. For each host, identify which slices of the GDA it needs to feed it's devices
  #       data_dim          model_dim (replicated)
  # [((slice(0, 2, None), slice(None, None, None)),...]
  local_slices = [(device, devices_to_slices[device]) for device in jax.local_devices()]
  # 4. Deduplicate these slices locally - each slice corresponds to a 'pipeline' required to load it
  slices, device_to_slice = deduplicate_slices(local_slices)

  #  5. Load the slices to the host (in this test, from an array - TODO: from .tfrecords)
  unique_host_buffers = load_slices_to_host(global_data, slices)

  # 6. Load them into the local device buffers and wrap it as one big GDA
  gda = construct_GDA(global_data_shape, global_mesh, data_axes, device_to_slice, unique_host_buffers)

  local_shards = [shard.data for shard in gda.local_shards]

  print(gda.local_data(0))
  print(gda.local_data(4))


test_one()


  