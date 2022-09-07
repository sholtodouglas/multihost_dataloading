import jax
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental import global_device_array as gda_lib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)

# TODO: Typing - e.g. for TfrtTpuDevices

_hashed_set_of_indexes = lambda indexes: hash(np.array([(v.start, v.stop) for idx in indexes for v in idx]).tobytes())

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

def deduplicate_indexes(local_indexes):
  '''
  [(slice(4, 6, None), slice(None, None, None)), (slice(4, 6, None), slice(None, None, None)), (slice(6, 8, None), slice(None, None, None)), (slice(6, 8│one, None, None)), (slice(2, 4, None), slice(None, None, None)), (slice(2, 4, None), slice(None, None, Non
  , None), ...]
  
  Returns the unique set of indexes we need to load locally
  And the mapping of device to those indexes
  '''
  unique_local_indexes = {}
  local_device_to_slice_hash = {}

  for (device, slice_tuple) in local_indexes:
    slice_hash = gda_lib._hashed_index(slice_tuple)
    if slice_hash not in unique_local_indexes:
      unique_local_indexes[slice_hash] = slice_tuple # d
    local_device_to_slice_hash[device] = slice_hash

  # unique_indexes : {((4, 6), (None, None)): (slice(4, 6, None), slice(None, None, None)), ((6, 8), (None, None)): (slice(6, 8, None), slice(None, None, None))}
  # device_to_slice_hash: [{TpuDevice(id=16, process_index=2, coords=(0,2,0), core_on_chip=0): 12334567,...]
  return unique_local_indexes, local_device_to_slice_hash

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

# instead of a million hash look ups, easier to just do equality checks

def test_per_replica_data_pipeline():
  '''
  We follow these steps:
  1. Initialise our desired GDA shape and device mesh layout
  2. Get the indexes of the GDA corresponding to each device
  3. For each host, identify which indexes it needs to load to feed it's devices
  4. Deduplicate these indexes
  5. Load the indexes (in this test, from an array - TODO: from .tfrecords)
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

# 2. Get the indexes of the GDA corresponding to each device
# returns e.g. [TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1):
#                                   (slice(6, 8, None), slice(None, None, None)),]
device_to_index = gda_lib.get_shard_indices(global_data_shape, global_mesh, data_axes)

# 3. For each host, identify which indexes of the GDA it needs to feed it's local devices
#       data_dim          model_dim (replicated)
# [((slice(0, 2, None), slice(None, None, None)),...]
local_indexes = [(device, device_to_index[device]) for device in jax.local_devices()]
# 4. Deduplicate these indexes locally - each slice corresponds to a 'pipeline' required to load it
unique_local_indexes, local_device_to_index_hash = deduplicate_indexes(local_indexes)

#  5. Load the slices to the host (in this test, from an array - TODO: from .tfrecords)
unique_host_buffers = load_slices_to_host(global_data, unique_local_indexes)

# 6. Load them into the local device buffers and wrap it as one big GDA
gda = construct_GDA(global_data_shape, global_mesh, data_axes, local_device_to_index_hash, unique_host_buffers)

local_shards = [shard.data for shard in gda.local_shards]

print(gda.local_data(0))
print(gda.local_data(4))

expected = np.split(global_data, 4, axis=0) # TODO: tidy
if jax.process_index == 0:
  assert gda.local_data(0) == expected[0]
  assert gda.local_data(4) == expected[1]
if jax.process_index == 2:
  assert gda.local_data(0) == expected[2]
  assert gda.local_data(4) == expected[3]

print("Option 3 '\u2713'")
  

###############################################################################################
######################## Per host data pipeline ###############################################
###############################################################################################

def get_total_length_of_unique_indexes(unique_indexes):
  size = 0
  for (data,model) in unique_indexes:
    size += data.stop - data.start
  return size


@dataclass
class Pipeline():
  hash: int
  size: int # contiguous length
  indices: Tuple[slice, slice]


def create_per_host_pipeline(host_devices_map, device_to_index_map):
  hash_to_unique_contiguous_global_indexes = {int: Pipeline}
  host_to_contiguous_global_index_hash = {}
  device_to_local_indexes = {}
  running_index = 0
  for host_id, host_devices in host_devices_map.items():
    #  [((slice(0, 2, None), slice(None, None, None)),...]
    host_device_to_global_indexes = [(device, device_to_index_map[device]) for device in host_devices]
    # deduplicate these
    index_hash_to_indexes_unique, device_to_slice_hash = deduplicate_indexes(host_device_to_global_indexes)
    unique_indexes = [v for _,v in index_hash_to_indexes_unique.items()]
    # hash the set of indices for this host
    pipeline_hash = _hashed_set_of_indexes(unique_indexes)
    host_to_contiguous_global_index_hash[host_id] = pipeline_hash
    pipeline_size = get_total_length_of_unique_indexes(unique_indexes)
    if pipeline_hash not in hash_to_unique_contiguous_global_indexes:
      # No model parallel slicing at the moment
      indices = (slice(running_index, running_index+pipeline_size, None), slice(None, None, None))
      running_index += pipeline_size
      pipeline = Pipeline(pipeline_hash, pipeline_size, indices)
      hash_to_unique_contiguous_global_indexes[pipeline_hash] = pipeline
    # NOTE: This does not allow for overlap of examples between hosts. It allows them to have the same examples 
    # (for model parallel hosts) but not different sets of overlap at a host level, we are determining how many 
    # unique pipelines there are and assigning each a contiguous slice of the data  within a host, we do the same 
    # - how many unique indexes are there, map each device to a slice hash and give them a contiguous slice of
    # the data loaded by that host
    data_per_device = pipeline_size // len(unique_indexes)
    slice_hash_to_pipeline_indices = {gda_lib._hashed_index(slice_tuple): \
                                                                        slice(i*data_per_device,(i+1)*data_per_device, None) \
                                                                          for i, slice_tuple in enumerate(unique_indexes)}
    # pipeline indices are the local indices of the data to be loaded by the pipeline
    for device, slice_hash in device_to_slice_hash.items():
      device_to_local_indexes[device] = slice_hash_to_pipeline_indices[slice_hash] 

  return hash_to_unique_contiguous_global_indexes, host_to_contiguous_global_index_hash, device_to_local_indexes


def load_pipeline(global_data, pipeline):
  '''
  In this example we load from an np array. 
  TODO: Load from a unique set of tfrecords.
  '''
  return global_data[pipeline.indices]

def test_per_host_data_pipeline():
  '''Test the case where we have one data pipeline per host.'''

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
  global_mesh = Mesh(test_mesh_layout, ('data', 'model'))
  # imagine these values are host id, and the numbers are each a device
  # we create four replicas, each split over two hosts
  #     00001111
  #     00001111
  #     22223333
  #     22223333
  
  # 2. Get the slices of the GDA corresponding to each device (globally)
  # returns e.g. [TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1):
  #                                   (slice(6, 8, None), slice(None, None, None)),]
  device_to_index = gda_lib.get_shard_indices(global_data_shape, global_mesh, data_axes)

  # What the t5x example does here is it splits the total length of the input data array
  # by num unique indicies and assigns each indices (or 'pipeline') to a host.
  # This doesn't account for the situation where multiple hosts might WANT to load the
  # same data. E.g in our case there are 4 unique slices ('pipelines'), but what we want
  # to test is each host loading the data required of it and not needing to reshard
  # during pjit. Instead, what we will do is:

  # 3. Remap the indices each host should load to one contiguous set per host
  hash_to_unique_contiguous_global_indexes, host_to_contiguous_global_index_hash, device_to_local_indexes = create_per_host_pipeline(host_devices_map, device_to_index)
  # hash_to_unique_contiguous_global_indexes: Objects representing slices like 0:4, 4:8  of the global data array
  # host_to_contiguous_global_index_hash: which contiguous section should each host load
  # device_to_local_indexes: of that contiguous section, which LOCAL indices go to which device

  # host_contiguous_indexes_to_load: which contiguous section should each host load
  host_contiguous_indexes_to_load = hash_to_unique_contiguous_global_indexes[host_to_contiguous_global_index_hash[jax.process_index()]]

  # End setup - begin iteration. Everything before this you only need to do once.
  # 4. Load that section
  local_data = load_pipeline(global_data, host_contiguous_indexes_to_load)

  # 5. Slice this up using local indcies and give it to the host local devices
  device_buffers = []
  for device in jax.local_devices():
    local_indices = device_to_local_indexes[device]
    data = local_data[local_indices]
    device_buffers.append(jax.device_put(data, device))

  # # 6. Load them into the local device buffers and wrap it as one big GDA
  gda = GlobalDeviceArray(global_data_shape, global_mesh, data_axes, device_buffers)
  # local_shards = [shard.data for shard in gda.local_shards]

  print(gda.local_data(0))
  print(gda.local_data(4))

  expected = np.split(global_data, 4, axis=0) # TODO: tidy
  if jax.process_index == 0:
    assert gda.local_data(0) == expected[0]
    assert gda.local_data(4) == expected[1]
  if jax.process_index == 2:
    assert gda.local_data(0) == expected[2]
    assert gda.local_data(4) == expected[3]

  print("Option 4 '\u2713'")

test_per_replica_data_pipeline()
test_per_host_data_pipeline()


  
# Note: tests which set up one host to have multiple processes https://source.corp.google.com/piper///depot/google3/learning/brain/research/jax/tests/tpu/multiprocess_tpu_test.py