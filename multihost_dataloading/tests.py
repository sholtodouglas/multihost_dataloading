import jax
from jax.experimental.maps import Mesh
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental import global_device_array as gda_lib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, NewType, Any, List, Dict

Device = NewType('Device', Any)  # TODO: update to the redef in t5x
Indexes = NewType('Device', Tuple[slice, slice])

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)

# TODO: Typing - e.g. for TfrtTpuDevices

_hashed_set_of_indexes = lambda indexes: hash(
    np.array([(v.start, v.stop) for idx in indexes for v in idx]).tobytes())


def device_to_host(devices: List[Device]):
  """Gets mapping from/to device ID <> host ID.

  Use the host->devices mapping to specify global device layouts that all hosts
  have visibility over (otherwise they would only have the indices of theirs).


  Args:
    devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0),
      core_on_chip=0),..]

  Returns:
    host_to_devices: {host_id: [device_1, device_2],..}
  """

  host_to_devices = defaultdict(list)
  for d in devices:
    host_to_devices[d.host_id].append(d)  # default dict so no need to check

  return host_to_devices


# TODO: Check this function and dehackify it
def construct_test_mesh_32(host_to_devices: Dict[int, List[Device]]):
  """By default, when we reshape a 32 (i.e.

  4 hosts, 8 devices each) slice to (data, model) unless the length of the
  model dimension is great than the number of devices per host, it will not
  be arranged with the second dimension crossing host boundaries.
  This makes sense - but we want to test the most general case: where a given
  host may have
  multiple independent model replicas on it (each loading different data), and
  where these
  replicas may stretch across host boundaries.

  This may occur! E.g. PaLM used 12 way model parallelism in a given replica.
  Therefore the layout may have been that one host had 8, the next 4 + 4, the
  next 8?

  To test this, we want a layout that looks like this, indices indicate host idx
  of devices

      00001111
      00001111
      22223333
      22223333

  Args:
    host_to_devices: {host_id: [device_1, device_2],..}
  """

  assert sum([len(devices) for _, devices in host_to_devices.items()]) == 32

  test_mesh_layout = np.array([
      host_to_devices[0][0:4] + host_to_devices[1][0:4],
      host_to_devices[0][4:8] + host_to_devices[1][4:8],
      host_to_devices[2][0:4] + host_to_devices[3][0:4],
      host_to_devices[2][4:8] + host_to_devices[3][4:8],
  ])

  return test_mesh_layout


def deduplicate_indexes(local_indexes: List[Tuple[Device, Tuple[slice,
                                                                slice]]]):
  """[(slice(4, 6, None), slice(None, None, None)),..]

  Returns the unique set of indexes we need to load locally
  And the mapping of device to those indexes
  """
  unique_local_indexes = {}
  local_device_to_index_hash = {}

  for (device, slice_tuple) in local_indexes:
    slice_hash = gda_lib._hashed_index(slice_tuple)
    if slice_hash not in unique_local_indexes:
      unique_local_indexes[slice_hash] = slice_tuple  # d
    local_device_to_index_hash[device] = slice_hash

  return unique_local_indexes, local_device_to_index_hash


def load_slices_to_host(global_data, pipelines: Dict[int, Tuple[slice, slice]]):
  """ For the moment just load from an array.

  TODO: Load from tf.data
  The device to pipeline hash will share the same hash here, so we
  can directly put from these buffers into local_device_buffers to
  form a GDA.
  """
  unique_host_buffers = {}
  for hash, slice in pipelines.items():
    unique_host_buffers[hash] = global_data[slice]
  return unique_host_buffers


# instead of a million hash look ups, easier to just do equality checks

################################################################################
######################## Per replica data pipeline #############################
################################################################################

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
  host_to_devices = device_to_host(devices)
  test_mesh_layout = construct_test_mesh_32(host_to_devices)
  # imagine these integers are host id, and the numbers are each a device
  # we create four replicas, each split over two hosts
  #     00001111
  #     00001111
  #     22223333
  #     22223333
  global_mesh = Mesh(test_mesh_layout, ('data', 'model'))

  # 2. Get the indexes of the GDA corresponding to each device
  # returns [TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1):
  #                               (slice(6, 8, None), slice(None, None, None)),]
  device_to_index = gda_lib.get_shard_indices(global_data_shape, global_mesh,
                                              data_axes)

  # 3. For each host, get indexes of the GDA it needs to feed it's local devices
  #       data_dim          model_dim (replicated)
  # [((slice(0, 2, None), slice(None, None, None)),...]
  local_indexes = [
      (device, device_to_index[device]) for device in jax.local_devices()
  ]
  # 4. Deduplicate indexes locally - each corresponds to a 'pipeline' used to load it
  unique_local_indexes, local_device_to_index_hash = deduplicate_indexes(
      local_indexes)

  # Below here is where you would loop
  #  5. Load the slices to the host  TODO: from .tfrecords)
  unique_host_buffers = load_slices_to_host(global_data, unique_local_indexes)

  # 6. Load them into the local device buffers and wrap it as one big GDA
  device_buffers = []
  for device, index_hash in local_device_to_index_hash.items():
    host_data = unique_host_buffers[index_hash]
    device_buffers.append(jax.device_put(host_data, device))

  gda = GlobalDeviceArray(global_data_shape, global_mesh, data_axes,
                          device_buffers)

  print(gda.local_data(0))
  print(gda.local_data(4))

  expected = np.split(global_data, 4, axis=0)  # TODO: tidy
  if jax.process_index == 0:
    assert gda.local_data(0) == expected[0]
    assert gda.local_data(4) == expected[1]
  if jax.process_index == 2:
    assert gda.local_data(0) == expected[2]
    assert gda.local_data(4) == expected[3]

  print("Option 3 '\u2713'")

################################################################################
######################## Per host data pipeline ################################
################################################################################


def get_total_length_of_unique_indexes(unique_indexes):
  size = 0
  for (data, _) in unique_indexes:
    size += data.stop - data.start
  return size


@dataclass
class Pipeline():
  hash: int
  size: int  # contiguous length
  contiguous_indices: Tuple[slice, slice]


def create_per_host_pipeline(host_to_devices: Dict[int, List[Device]],
                             device_to_index: Dict[Device, Tuple[slice,
                                                                 slice]]):
  '''
  1. Get which sets of indices of data each host needs to feed its devices
  2. Determine how many unique sets of these indices there are (e.g. in our case
      there are 2)
  3. Assign each a contiguous section of the data = to it's length (this accounts
      for if they were previosuly non-contiguous)
  4. Determine the local device's indexes into this contiguous section for each host
      
  NOTE: This does not allow for incomplete overlap of examples between hosts. 
  It allows them to have the same examples.

  Returns:
    hash_to_unique_pipeline: Pipelines representing slices like 0:4, 4:8 of the
    global data array - eachshould have a unique tf.data pipeline set up.
    host_to_pipeline_hash: Which pipeline should each host use to load it's data
    device_to_local_indexes: Of the data loaded by that pipeline, which local 
    indices go to which device

  '''
  hash_to_unique_pipeline = {int: Pipeline}
  host_to_pipeline_hash = {int: int}
  device_to_local_indexes = {Device: Tuple[slice, slice]}
  running_index = 0
  for host_id, host_devices in host_to_devices.items():
    #  [((slice(0, 2, None), slice(None, None, None)),...]
    host_device_to_global_indexes = [
        (device, device_to_index[device]) for device in host_devices
    ]
    # deduplicate these
    index_hash_to_indexes_unique, device_to_index_hash = deduplicate_indexes(
        host_device_to_global_indexes)
    unique_indexes = [v for _, v in index_hash_to_indexes_unique.items()]
    # hash the set of indices for this host
    pipeline_hash = _hashed_set_of_indexes(unique_indexes)
    host_to_pipeline_hash[host_id] = pipeline_hash
    pipeline_size = get_total_length_of_unique_indexes(unique_indexes)
    if pipeline_hash not in hash_to_unique_pipeline:
      # No model parallel slicing at the moment
      indices = (slice(running_index, running_index + pipeline_size,
                       None), slice(None, None, None))
      running_index += pipeline_size
      pipeline = Pipeline(pipeline_hash, pipeline_size, indices)
      hash_to_unique_pipeline[pipeline_hash] = pipeline

    # within a host, we do the same - how many unique indexes are there,
    # map each device to a index hash and give them a subsection of the data
    # loaded by that host
    data_per_device = pipeline_size // len(unique_indexes)
    slice_hash_to_pipeline_indices = {
        gda_lib._hashed_index(slice_tuple):
        slice(i * data_per_device, (i + 1) * data_per_device, None)
        for i, slice_tuple in enumerate(unique_indexes)
    }
    # pipeline indices are the local indices of the data to be loaded by the pipeline
    for local_device, slice_hash in device_to_index_hash.items():
      device_to_local_indexes[local_device] = slice_hash_to_pipeline_indices[
          slice_hash]

  return hash_to_unique_pipeline, host_to_pipeline_hash, device_to_local_indexes


def load_pipeline(global_data, pipeline: Pipeline):
  """In this example we load from an np array.

  TODO: Load from a unique set of tfrecords.
  """
  return global_data[pipeline.contiguous_indices]


def test_per_host_data_pipeline():
  """Test the case where we have one data pipeline per host."""

  # 1. Initialise our desired GDA shape and deivce mesh layout
  # Construct global data
  global_data_shape = (8, 4)
  global_data = np.arange(np.prod(global_data_shape)).reshape(global_data_shape)
  data_axes = P('data', None)
  # Create our device mesh - this function arranges a 4 hosts/32 devices
  # to allow us to test the general case
  devices = jax.devices()
  host_to_devices = device_to_host(devices)
  test_mesh_layout = construct_test_mesh_32(host_to_devices)
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

  # What the t5x example does here is it splits the total length of the input
  # data array by num unique indicies and assigns each indices (or 'pipeline')
  # to a host. This doesn't account for the situation where multiple hosts might
  # WANT to load the same data. E.g in our case there are 4 unique slices
  # ('pipelines'), but what we want to test is each host loading the data
  # required of it and not needing to reshard during pjit. Instead, we

  # 3. Remap the indices each host should load to one contiguous set per host
  hash_to_unique_pipeline, host_to_pipeline_hash, device_to_local_indexes = create_per_host_pipeline(
      host_to_devices, device_to_index)
  # hash_to_unique_pipeline: Pipelines representing slices like 0:4, 4:8  of the
  # global data array - each  should have a unique tf.data pipeline set up.
  # host_to_pipeline_hash: Which pipeline should each host use to load it's data
  # device_to_local_indexes: Of the data loaded by that pipeline, which local
  # indices go to which device

  # host_pipeline_to_load: which contiguous section should each host load
  host_pipeline_to_load = hash_to_unique_pipeline[host_to_pipeline_hash[
      jax.process_index()]]

  # End setup - begin iteration. 
  # 4. Load that section
  local_data = load_pipeline(global_data, host_pipeline_to_load)

  # 5. Slice this up using local indices and give it to the host local devices
  device_buffers = []
  for device in jax.local_devices():
    local_indices = device_to_local_indexes[device]
    data = local_data[local_indices]
    device_buffers.append(jax.device_put(data, device))

  # # 6. Load them into the local device buffers and wrap it as one big GDA
  gda = GlobalDeviceArray(global_data_shape, global_mesh, data_axes,
                          device_buffers)
  # local_shards = [shard.data for shard in gda.local_shards]

  print(gda.local_data(0))
  print(gda.local_data(4))

  expected = np.split(global_data, 4, axis=0)  # TODO: tidy
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
