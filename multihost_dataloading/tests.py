from functools import total_ordering
import jax
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental import global_device_array as gda_lib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, NewType, Any, List, Dict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable CUDA not found etc warnings
import tensorflow as tf

# TODO: Typing - e.g. for TfrtTpuDevices
Device = NewType('Device', Any)  # TODO: update to the redef in t5x
DeviceBuffer = NewType('DeviceBuffer', Any)  # TODO: update to the redef in t5x

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)

data_dim = 0 # assume data dimension is the first 

def construct_test_mesh_32() -> np.ndarray:
  """Constructs a non-standard mesh layout to test all cases.

  By default, when we reshape a 32 (i.e. 4 hosts, 8 devices each) slice to (data, model) 
  unless the length of the model dimension is great than the number of devices per host, 
  it will not be arranged with the second dimension crossing host boundaries.
  This makes sense - but we want to test the most general case: where a given
  host may have multiple independent model replicas on it (each loading different data),
  and where these replicas may stretch across host boundaries.

  This may occur! E.g. PaLM used 12 way model parallelism in a given replica.
  Therefore the layout may have been that one host had 8, the next 4 + 4, the
  next 8?

  To test this, we want a layout that looks like this, values indicate host idx
  of devices

      00001111
      00001111
      22223333
      22223333

  Returns:
    test_mesh_layout: np.ndarray of Device objects
  """
  host_to_devices = defaultdict(list)
  for d in jax.devices(): host_to_devices[d.host_id].append(d)  # default dict so no need to check

  assert sum([len(devices) for _, devices in host_to_devices.items()]) == 32

  test_mesh_layout = np.array([
      host_to_devices[0][0:4] + host_to_devices[1][0:4],
      host_to_devices[0][4:8] + host_to_devices[1][4:8],
      host_to_devices[2][0:4] + host_to_devices[3][0:4],
      host_to_devices[2][4:8] + host_to_devices[3][4:8],
  ])

  return test_mesh_layout

################################################################################
######################## Per replica data pipeline #############################
################################################################################

@dataclass
class shardInfo:
  idx: int
  size: int 

def get_per_replica_data_pipeline(dataset: tf.data.Dataset, 
                device_to_index: Dict[Device, Tuple[slice, slice]]) -> Tuple[Dict[Device, shardInfo], Dict[int, tf.data.Dataset]]:
  '''Create a tf.dataset per unique slice of data to be loaded to the host (i.e per replica).'''

  # get the unique set of slices into the GDA (i.e one per replica)
  # and which devices map to those
  index_hash_to_shard_idx = {} # int, int
  device_to_shard_info = {} # device, (int, 
  for (device, index_tuple) in device_to_index.items():
    index_hash = gda_lib._hashed_index(index_tuple)

    if index_hash not in index_hash_to_shard_idx:
      index_hash_to_shard_idx[index_hash] = len(index_hash_to_shard_idx)

    indices_size = index_tuple[data_dim].stop - index_tuple[data_dim].start
    device_to_shard_info[device] = shardInfo(index_hash_to_shard_idx[index_hash], indices_size)

  num_shards = len(index_hash_to_shard_idx)
  
  # For each host, get the dataset shards for it's local devices
  # and map the devices to those shards. 
  shard_idx_to_dataset = {}
  for device in jax.local_devices():
    shard_info = device_to_shard_info[device]

    sharded_dataset = iter(dataset
                        .shard(num_shards=num_shards,index=shard_info.idx)
                        .batch(shard_info.size)
                        .repeat()
                        .as_numpy_iterator()) # for Jax
    
    # we only want one copy of each pipeline per host
    if shard_info.idx not in shard_idx_to_dataset:
      shard_idx_to_dataset[shard_info.idx] = sharded_dataset

  return device_to_shard_info, shard_idx_to_dataset

def get_next_per_replica(device_to_shard_info: Dict[Device, shardInfo], shard_idx_to_dataset: Dict[int, tf.data.Dataset]) -> List[DeviceBuffer]:
  # load one iteration of each of those datasets
  shard_idx_to_loaded_data = {idx: dataset.next() for idx, dataset in shard_idx_to_dataset.items()}
  # iterate over the local devices, getting the copy of preloaded data
  device_buffers = []
  for device in jax.local_devices():
    data_shard_info = device_to_shard_info[device]
    data = shard_idx_to_loaded_data[data_shard_info.idx]
    device_buffers.append(jax.device_put(data, device))
  
  return device_buffers
################################################################################
######################## Per host data pipeline ################################
################################################################################

_hashed_set_of_indexes = lambda indexes: hash(
    np.array([(v.start, v.stop) for idx in indexes for v in idx]).tobytes())

def get_total_length_of_unique_indexes(unique_indexes: List[Tuple[slice, slice]]) -> int:
  size = 0
  for (data, _) in unique_indexes:
    size += data.stop - data.start
  return size
  
def deduplicate_indexes(local_indexes: List[Tuple[Device, Tuple[slice,
                                                                slice]]]) -> Dict[int, Tuple[slice, slice]]:
  """Returns the unique set of indexes we need to load locally.
  Uses the indices hash to check for sameness.
  """
  unique_indices = {} # hash, indices

  for (_, index_tuple) in local_indexes:
    index_hash = gda_lib._hashed_index(index_tuple)
    if index_hash not in unique_indices:
      unique_indices[index_hash] = index_tuple  

  return unique_indices

def get_unique_shards(host_to_devices: Dict[int, List[Device]],
                             device_to_index: Dict[Device, Tuple[slice,
                                                                 slice]]) -> Tuple[Dict[int, int], Dict[int, int]]:
  '''Looks at the sets of data each host needs, deduplicates, assigns a shard to the set.'''

  host_to_dataset_shard = {} # [process_id, index]
  dataset_shard_hash_to_index = {} # [hash, index]

  for host_id, host_devices in host_to_devices.items():

    # get exclusively the indexes for host_id
    host_device_to_global_indexes = [
        (device, device_to_index[device]) for device in host_devices
    ]
    # deduplicate these
    idx_hash_to_unique_indices= deduplicate_indexes(host_device_to_global_indexes)

    # hash the set of indices for this host
    pipeline_hash = _hashed_set_of_indexes([idx for _, idx in idx_hash_to_unique_indices.items()])

    # assign each hosts' (set of indices) a shard index in the order we discover them
    # this will be the shard index loaded by tf.data 
    if pipeline_hash not in dataset_shard_hash_to_index:
      dataset_shard_hash_to_index[pipeline_hash] = len(dataset_shard_hash_to_index)

    host_to_dataset_shard[host_id] = dataset_shard_hash_to_index[pipeline_hash]
  
  # tf.data requires total num shards
  num_unique_shards = len(dataset_shard_hash_to_index)
  return host_to_dataset_shard, num_unique_shards

def convert_global_indices_to_local_indices(device_to_index: Dict[Device, Tuple[slice,
                                                                 slice]]) -> Tuple[Dict[Device, slice], int]:
  '''Converts global GDA indices for each device to local indices of host loaded data.'''
  device_index_hash_to_local_index = {} # hash, [slice, slice]
  device_to_local_indices = {} # device, [slice, slice]
  total_data_to_load = 0
  for device in jax.local_devices():
    # get a hash to identify this unique slice into the global array
    global_index_tuple = device_to_index[device]
    global_index_hash = gda_lib._hashed_index(global_index_tuple)
    if global_index_hash not in device_index_hash_to_local_index:
      # assign the global slice a slice from the local data
      # it doesn't matter if this was the same as the original slice
      # as long as it is the same size, and shard among other devices 
      # who match that global slice hash

      # TODO: No magic indexing
      size = global_index_tuple[data_dim].stop - global_index_tuple[data_dim].start
      local_slice = slice(total_data_to_load, total_data_to_load+size, None)
      device_index_hash_to_local_index[global_index_hash] = local_slice
      total_data_to_load += size

    device_to_local_indices[device] = device_index_hash_to_local_index[global_index_hash]

  return device_to_local_indices, total_data_to_load


def get_per_host_data_pipeline(dataset: tf.data.Dataset, device_to_index: Dict[Device, Tuple[slice, slice]]) -> Tuple[tf.data.Dataset, Dict[Device, slice]]:
  """Test the case where we have one data pipeline per host."""

  host_to_devices = defaultdict(list)
  for d in jax.devices(): host_to_devices[d.host_id].append(d)  # default dict so no need to check
  # Now, we want to find the number of unique (per host) dataset shards which should be loaded
  # and assign each host to their shard.
  # TODO: Check if this is even necessary, or we could get as good results with interleave
  host_to_dataset_shard, num_shards = get_unique_shards(host_to_devices, device_to_index)
  # And assign devices indices into the data to be loaded by the host
  host_local_indices, total_data_to_load = convert_global_indices_to_local_indices(device_to_index)

  # Create the data pipeline
  local_data_shard_index = host_to_dataset_shard[jax.process_index()]
  sharded_dataset = iter(dataset
                        .shard(num_shards=num_shards,index=local_data_shard_index)
                        .batch(total_data_to_load)
                        .repeat()
                        .as_numpy_iterator()) # for Jax

  return sharded_dataset, host_local_indices 

def get_next_per_host(sharded_dataset: tf.data.Dataset, host_local_indices: Dict[Device, slice]) -> List[DeviceBuffer]:
    
    # load a single pipeline for the entire host
    local_data = sharded_dataset.next()
    # Slice this up using local indices and give it to the host local devices
    device_buffers = []
    for device in jax.local_devices():
      local_indices = host_local_indices[device]
      data = local_data[local_indices]
      device_buffers.append(jax.device_put(data, device))
    
    return device_buffers


################################################################################
######################## Universal code ########################################
################################################################################


def test_case(method: str):
  # 1. Initialise our desired GDA shape and deivce mesh layout
  # Construct global data
  global_data_shape = (8, 4)
  global_data = np.arange(np.prod(global_data_shape)).reshape(global_data_shape)
  data_axes = P('data', None)
  # Create our device mesh - this function arranges a 4 hosts/32 devices
  # to allow us to test the general case
  test_mesh_layout = construct_test_mesh_32()
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
  device_to_index = gda_lib.get_shard_indices(global_data_shape, global_mesh,
                                              data_axes)                                 
  dataset = tf.data.Dataset.from_tensor_slices(global_data)

  if method == 'per_replica':
    # get the mapping of devices to data shards, and the unique set of shard indices with their 
    # corresponding datasets
    device_to_shard_info, shard_idx_to_dataset = get_per_replica_data_pipeline(dataset, device_to_index)
    device_buffers = get_next_per_replica(device_to_shard_info, shard_idx_to_dataset)
    
  elif method == 'per_host':
    # returns a single tf.data shard per host, and how the local devices index into this
    sharded_dataset, host_local_indices = get_per_host_data_pipeline(dataset, device_to_index)
    device_buffers = get_next_per_host(sharded_dataset, host_local_indices)

  else: 
    raise NotImplementedError

  ##################################  End setup, start iteration ################################

  # # 6. Load them into the local device buffers and wrap it as one big GDA
  gda = GlobalDeviceArray(global_data_shape, global_mesh, data_axes,
                          device_buffers)

  print(gda.local_data(0))
  print(gda.local_data(4))

  test_gda_output(global_data, gda, method)

def test_gda_output(global_data: np.ndarray, gda: GlobalDeviceArray, method: str):
  if method == 'per_replica':
    # Per host re-indexes s.t. there is only one tf.data shard per host 
    # TODO: Fix magic numbers
    if jax.process_index() == 0 or jax.process_index() == 1:
      assert (gda.local_data(0) == global_data[0::4]).all()
      assert (gda.local_data(4) == global_data[1::4]).all()
    if jax.process_index() == 2 or jax.process_index() == 3:
      assert (gda.local_data(0) == global_data[2::4]).all()
      assert (gda.local_data(4) == global_data[3::4]).all()
    
  elif method == 'per_host':
    # Per host re-indexes s.t. there is only one tf.data shard per host 
    # TODO: Fix magic numbers
    if jax.process_index() == 0 or jax.process_index() == 1:
      assert (gda.local_data(0) == global_data[0::2][:2]).all()
      assert (gda.local_data(4) == global_data[0::2][2:]).all()
    if jax.process_index() == 2 or jax.process_index() == 3:
      assert (gda.local_data(0) == global_data[1::2][:2]).all()
      assert (gda.local_data(4) == global_data[1::2][2:]).all()
    
  else:
    raise NotImplementedError
    

  print(f"Method: {method} '\u2713'")


test_case('per_host')
test_case('per_replica')


# Note: tests which set up one host to have multiple processes
#  https://source.corp.google.com/piper///depot/google3/learning/brain/research/jax/tests/tpu/multiprocess_tpu_test.py
