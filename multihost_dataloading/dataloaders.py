"""Code to test data pipeline arrangements.
Code arranged as follows
- Initalisation code
- Per test case unique code
  - For each, we have a factory function which returns a 'get_next_gda' fn.
    This factory accepts global_data_shape and data_axes as pytrees because
    datasets are often returned by the .next call as a dictionary or tuple,
    e.g. representing inputs and labels. As a result, the get_next_gda fn
    returns a pytree of gdas. This necessitates a little fancy tree_mapping 
    internally.
- Test harness
TODO(sholto):
- Properly benchmark
"""

from collections import defaultdict  # pylint: disable=g-importing-member
from dataclasses import dataclass  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
import os
from typing import Callable, Any, Dict, List, Tuple

import jax
from jax.experimental import global_device_array as gda_lib
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import Device
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from jax.experimental.pjit import with_sharding_constraint
import jax.numpy as jnp
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable CUDA not found etc warnings
import tensorflow as tf  # pylint: disable=g-import-not-at-top

# make pjit output GDAs
jax.config.update('jax_parallel_functions_output_gda', True)

Pytree = Any

data_dim = 0  # assume data dimension is the first


def construct_test_mesh_32() -> np.ndarray:
  """Constructs a non-standard mesh layout to test all cases.
  By default, when we reshape a 32 (i.e. 4 hosts, 8 devices each) slice to
  (data, model) unless the length of the model dimension is great than the
  number of devices per host, it will not be arranged with the second dimension
  crossing host boundaries. This makes sense - but we want to test the most
  general case: where a given host may have multiple independent model replicas
  on it (each loading different data), and where these replicas may stretch
  across host boundaries.
  This may occur! E.g. PaLM used 12 way model parallelism in a given replica.
  Which is not evenly divisble by the number of devices on many TPU platforms.
  To test this, we want a layout that looks like this, values indicate host idx
  of devices.
      00001111
      00001111
      22223333
      22223333
  Returns:
    test_mesh_layout: np.ndarray of Device objects
  """
  host_to_devices = defaultdict(list)
  for d in jax.devices():
    host_to_devices[d.host_id].append(d)  # default dict so no need to check

  assert sum([len(devices) for _, devices in host_to_devices.items()]) == 32

  test_mesh_layout = np.array([
      host_to_devices[0][0:4] + host_to_devices[1][0:4],
      host_to_devices[0][4:8] + host_to_devices[1][4:8],
      host_to_devices[2][0:4] + host_to_devices[3][0:4],
      host_to_devices[2][4:8] + host_to_devices[3][4:8],
  ])

  return test_mesh_layout


def check_inputs(dataset, global_data_shape, data_axes):

  # TODO(sholto): Is there a way to do this without calling dataset?
  dataset_structure = jax.tree_util.tree_structure(iter(dataset).next())
  global_data_shape_structure = jax.tree_util.tree_structure(global_data_shape)
  data_axes_structure = jax.tree_util.tree_structure(data_axes)

  try:
    assert dataset_structure == global_data_shape_structure == data_axes_structure, 'All inputs should have the same pytree structure.'
  except AssertionError as msg:
    (print(
        f"""{msg} - The most likely reason for this is that global shapes should
        be arrays or claases not tuples - otherwise tree map enumerates indiviudal
        dimensions as leaves. Dataset: {dataset_structure}, \n Shapes:
          {global_data_shape_structure}, \n Axes: {data_axes_structure}"""))

  shapes, _ = jax.tree_util.tree_flatten(global_data_shape)
  batch_dims = [s[0] for s in shapes]



  assert all(b == batch_dims[0]for b in batch_dims), 'All batch axis should be equal for gdas'

  assert all(b[0] == shapes[0][0] for b in shapes), 'All dataset elements should be sharded along the data axis identically'

  batch_dim = batch_dims[0]
  return batch_dim


################################################################################
################### (Strawman) Load all data on all hosts ######################
################################################################################


def get_all_data_all_hosts_pipeline(dataset: tf.data.Dataset,
                                    global_data_shape: Pytree,
                                    global_mesh: Mesh,
                                    data_axes: Pytree) -> Callable[[], Pytree]:
  """Return the same, globally sized dataloader across all hosts."""

  batch_dim = check_inputs(dataset, global_data_shape, data_axes)

  # Get the slices of the GDA corresponding to each device (globally)
  # returns [TpuDevice(id=27, process_index=2, coords=(1,3,0), core_on_chip=1):
  #                              (slice(6, 8, None), slice(None, None, None)),]
  # this is now a pytree of 'device_to_index' objects
  # matching the structure of dataloader outputs
  device_to_index = jax.tree_map(
    lambda shape, axes: gda_lib.get_shard_indices(shape, global_mesh, axes),
    global_data_shape,
    data_axes)

  # one data pipeline, the same across every host
  dataset = (dataset.batch(batch_dim).repeat().as_numpy_iterator())

  next_fn = partial(
      get_next_all_data_all_hosts,
      dataset,
      device_to_index,
      global_data_shape,
      global_mesh,
      data_axes,
  )
  return next_fn


def get_next_all_data_all_hosts(dataset, device_to_index: Dict[Device,
                                                               Tuple[slice,
                                                                     slice]],
                                global_data_shape: Pytree, global_mesh: Mesh,
                                data_axes: P) -> Pytree:
  """Fill device buffers with appropriate slice of the globally identical data."""
  batch = dataset.next()

  # for each leaf in the data output pytree
  def form_gda(element, shape, axes, device_to_index):
    # iterate over the local devices, getting the correct slice
    device_buffers = [
        jax.device_put(element[device_to_index[device]], device)
        for device in jax.local_devices()
    ]
    #  Wrap device buffers as GDA
    gda = GlobalDeviceArray(shape, global_mesh, axes, device_buffers)
    return gda

  # tree map over multiple trees uses the first tree as a structure 
  # prefix, which means subsequent trees follow
  pytree_of_gdas = jax.tree_map(
      form_gda, batch, global_data_shape, data_axes, device_to_index)

  return pytree_of_gdas


################################################################################
######################## Per replica data pipeline #############################
################################################################################


@dataclass
class ShardInfo:
  idx: int
  size: int


def get_per_replica_data_pipeline(
    dataset: tf.data.Dataset, global_data_shape: Pytree, global_mesh: Mesh,
    data_axes: Pytree) -> Callable[[], GlobalDeviceArray]:
  """Create a tf.dataset per unique slice of data to be loaded to the host (i.e per replica).
  Identifies what data the host wants to load for it's devices, deduplicates it
  - and returns a data pipeline per unique slice desired by the devices. In 'get
  next', it sequentially loads each of these. This is simpler than the per host
  method, but introduces overheads from several sequential calls to a data
  pipeline, as opposed to a single equivalently sized call.
  Returns two dicts because we only want to load each of the datasets in
  shard_idx_to_dataset once, not once per device - so we need both the
  unique pipelines, and the per device mapping to them.
  + Efficiently deduplicates the data to load per host
  + Low-medium complexity
  - Overhead from multiple calls to tf.data
  Args:
    dataset: tf dataset over all files
    global_data_shape: what the size of the GDA should be
    global_mesh: global deivces mesh
    data_axes: axes along which data is partitioned
  Returns:
    next_fn: Function to get the next batch as a pytree of GDAs
  """

  check_inputs(dataset, global_data_shape, data_axes)

  # this is now a pytree of 'device_to_index' objects
  # matching the structure of dataloader outputs
  device_to_index = jax.tree_map(
    lambda shape, axes: gda_lib.get_shard_indices(shape, global_mesh, axes),
    global_data_shape,
    data_axes)

  # We want two things out of this next piece of code.
  # device_to_shard_info: Which shard of data should go on which
  # device. This should be a pytree of the same structure as data.
  # Because a given shard of data will load all of the shards of
  # data_pipeline_pytree form.

  # Therefore, we place this within the pytree function.

  # shard_idx_to_dataset: Which idx corresponds to which dataset
  # pipeline. However, as we know that the partitioning along the
  # data dimension of each dataset element is the same, and the dataset
  # loads all elements in one, we only need one of this object.

  # Therefore, we place this outside the pytree function.

  # For each host, get the dataset shards for it's local devices
  # and map the devices to those shards.
  shard_idx_to_dataset = {}

  def identify_shards(_, device_to_index) -> Dict[Device, ShardInfo]:
    # Now, the following gets the unique set of slices into the
    # GDA (i.e one per replica) and which devices map to those.
    index_hash_to_shard_idx : Dict[int, int] = {}
    device_to_shard_info : Dict[Device, int] = {}
    for (device, index_tuple) in device_to_index.items():
      index_hash = gda_lib._hashed_index(index_tuple)  # pylint: disable=protected-access
      shard_idx = index_hash_to_shard_idx.setdefault(index_hash, len(index_hash_to_shard_idx))
      indices_size = index_tuple[data_dim].stop - index_tuple[data_dim].start
      device_to_shard_info[device] = ShardInfo(shard_idx, indices_size)

    num_shards = len(index_hash_to_shard_idx)
    for device in jax.local_devices():
      shard_info = device_to_shard_info[device]
      if shard_info.idx not in shard_idx_to_dataset:
        shard_idx_to_dataset[shard_info.idx] = iter(
            dataset.shard(num_shards=num_shards, index=shard_info.idx)
                  .batch(shard_info.size)
                  .repeat()
                  .as_numpy_iterator())

    return device_to_shard_info

  # create one of these for each dataset element
  # so that it can slice into the generated data appropriately
  device_to_shard_info = jax.tree_map(identify_shards, global_data_shape, device_to_index)

  next_fn = partial(
      get_next_per_replica,
      device_to_shard_info,
      shard_idx_to_dataset,
      global_data_shape,
      global_mesh,
      data_axes,
  )

  return next_fn


def transpose_and_wrap_per_shard(
    shard_idx_to_loaded_data: Dict[int, Pytree]) -> Pytree:
  """Creates necessary datastructure to provide dicts of shards to tree_map.
    After calling data.next() for each shard, we now have arbitrary pytrees
    under each shard idx as outputs from the dataloader. E.g. we might have
    the following, with a nested multilevel dict/tuple.
    {shard_idx1: {'inputs': (Array, Array) 'labels': Array}, shard_idx2: ...}
    we want instead
    {'inputs': ({shard_idx1: Array, shard_idx2: Array}, ...), 'labels': ...}
    where container is a dict containing all the shards for that object to be
    formed into a GDA, but a standard dict would be enumerated by tree_map
    rather than passed whole, so use a custom unregistered "leaf-like" dict
    type instead.
    Args:
      shard_index_to_loaded_data: A dict mapping shard indices to that shard of
      tf.data results
    Returns:
      A pytree matching an individual tf.data call, but holding a custom dict
      as each leaf so that a GDA can be formed from sharded pieces.
  """
  outer_structure = jax.tree_util.tree_structure(
      {k: 0 for k in shard_idx_to_loaded_data})
  inner_structure = jax.tree_util.tree_structure(
      next(iter(shard_idx_to_loaded_data.values())))
  transposed_tree = jax.tree_util.tree_transpose(
      outer_structure,
      inner_structure,
      shard_idx_to_loaded_data)
  return transposed_tree


def get_next_per_replica(device_to_shard_info: Pytree,
                         shard_idx_to_dataset: Dict[int, tf.data.Dataset],
                         global_data_shape: Pytree, global_mesh: Mesh,
                         data_axes: Pytree) -> GlobalDeviceArray:
  """Gets the next batch of filled device_buffers using per replica pipelines."""
  # load one iteration of each of those datasets
  shard_idx_to_loaded_data = {
      idx: dataset.next() for idx, dataset in shard_idx_to_dataset.items()
  }

  per_output_sharded_info = transpose_and_wrap_per_shard(
      shard_idx_to_loaded_data)

  # for each leaf in the data output pytree
  # pass shape first as global_data_shape defines tree structure prefix
  def form_gda(shape: Pytree,
               per_output_sharded_info: Dict[int, tf.data.Dataset],
               device_to_shard_info: Dict[Device, ShardInfo],
               axes: Pytree) -> GlobalDeviceArray:

    device_buffers = []
    for device in jax.local_devices():
      data_shard_info = device_to_shard_info[device]
      data = per_output_sharded_info[data_shard_info.idx]
      device_buffers.append(jax.device_put(data, device))
    #  Wrap device buffers as GDA
    gda = GlobalDeviceArray(shape, global_mesh, axes, device_buffers)
    return gda

  pytree_of_gdas = jax.tree_map(form_gda,
                                global_data_shape,
                                per_output_sharded_info,
                                device_to_shard_info,
                                data_axes)
  return pytree_of_gdas


################################################################################
######################## Per host data pipeline ################################
################################################################################


def get_unique_shards(
    host_to_devices: Dict[int, List[Device]],
    device_to_index: Dict[Device, Tuple[slice, slice]]
) -> Tuple[Dict[int, int], Dict[int, int]]:
  """Looks at the sets of data each host needs, deduplicates, assigns a shard to the set."""

  host_to_dataset_shard = {}  # [process_id, index]
  dataset_shard_hash_to_index = {}  # [hash, index]

  for host_id, host_devices in host_to_devices.items():
    host_indices = [device_to_index[device] for device in host_devices]
    hashable_indices = jax.tree_map(lambda s: (s.start, s.stop), host_indices)
    pipeline_hash = hash(tuple(set(hashable_indices)))
    # assign each host's set of indices a shard index in the order we discover
    # this will be the shard index loaded by tf.data
    host_to_dataset_shard[host_id] = dataset_shard_hash_to_index.setdefault(
        pipeline_hash, len(dataset_shard_hash_to_index))

  # tf.data requires total num shards
  num_unique_shards = len(dataset_shard_hash_to_index)
  return host_to_dataset_shard, num_unique_shards


def convert_global_indices_to_local_indices(
    device_to_index: Dict[Device, Tuple[slice, slice]]
) -> Tuple[Dict[Device, slice], int]:
  """Converts global GDA indices for each device to local indices of host loaded data."""

  local_indices = [device_to_index[device] for device in jax.local_devices()]
  # Tacit assumption that we -only- shard dataset batch along data dim here, we could
  # relax this but I'm not sure it would actually be handled right by this approach:
  data_indices = [(s[data_dim].start, s[data_dim].stop) for s in local_indices]
  unique_slice_sizes = {idx: idx[1]-idx[0] for idx in data_indices}

  # assign a unique local data slice to each device
  total_data_to_load = 0
  device_index_hash_to_local_index = {}
  for idx, size in unique_slice_sizes.items():
    device_index_hash_to_local_index[idx] = slice(total_data_to_load, total_data_to_load + size)
    total_data_to_load += size

  device_to_local_indices = {}
  for device, data_index in zip(jax.local_devices(), data_indices):
    device_to_local_indices[device] = device_index_hash_to_local_index[data_index]

  return device_to_local_indices, total_data_to_load


def get_per_host_data_pipeline(dataset: tf.data.Dataset,
                               global_data_shape: np.ndarray, global_mesh: Mesh,
                               data_axes: P) -> Callable[[], Pytree]:
  """Test the case where we have one data pipeline per host.
  To do this, we determine which pieces of data each host needs to feed it's
  devices,
  identify the unique sets of these (which is likely < num_hosts), and then
  create
  a data pipeline for each set.
  + No overhead from multiple pipelines per host
  - High complexity
  - Doesn't allow for incomplete overlap in the batches loaded by hosts
  Args:
    dataset: tf dataset over all files
    global_data_shape: what the size of the GDA should be
    global_mesh: global deivces mesh
    data_axes: axes along which data is partitioned
  Returns:
    sharded_dataset: Correct dataset to load for this host
    host_local_indices: indices for just the data loaded by the host's pipeline
  """

  check_inputs(dataset, global_data_shape, data_axes)

  # pytree of 'device_to_index' objects matching the structure of data
  device_to_index = jax.tree_map(
    lambda shape, axes: gda_lib.get_shard_indices(shape, global_mesh, axes),
    global_data_shape,
    data_axes)

  # group by host_id
  host_to_devices = defaultdict(list)
  for d in jax.devices():
    host_to_devices[d.host_id].append(d)

  # Now, we want to find the number of unique (per host) dataset shards which
  # should be loaded and assign each host to their shard.

  # Now, as we are creating our own slice in this function, and assuming that
  # we only have one dimension we are sharding along, we don't need to do
  # clever tree mapping as the unique shards -> therefore just take
  # the first one and get the unique sharding from that.
  dataset_structure = jax.tree_util.tree_structure(global_data_shape)
  representative_device_to_index = dataset_structure.flatten_up_to(
      device_to_index)[0]
  host_to_dataset_shard, num_shards = get_unique_shards(
    host_to_devices, representative_device_to_index)
  # And assign devices indices into the data to be loaded by the host
  # The slices generated here are only along the batch dim, and thus will work
  # for all items in the data output pytree
  host_local_indices, total_data_to_load = convert_global_indices_to_local_indices(
      representative_device_to_index)

  # Create the data pipeline
  local_data_shard_index = host_to_dataset_shard[jax.process_index()]
  sharded_dataset = iter(
      dataset.shard(num_shards=num_shards, index=local_data_shard_index)
             .batch(total_data_to_load)
             .repeat()
             .as_numpy_iterator())

  next_fn = partial(
      get_next_per_host,
      sharded_dataset,
      host_local_indices,
      global_data_shape,
      global_mesh,
      data_axes,
  )

  return next_fn


def get_next_per_host(sharded_dataset: tf.data.Dataset,
                      host_local_indices: Dict[Device, slice],
                      global_data_shape: Pytree, global_mesh: Mesh,
                      data_axes: P) -> GlobalDeviceArray:
  """Get device buffers to form GDA using per host pipeline."""

  # load from a single pipeline for the entire host
  # this is returned as a pytree in the same shape as global data shape
  local_data = sharded_dataset.next()
  # Slice this up using local indices and give it to the host local devices
  def form_gda(element, shape, axes) -> GlobalDeviceArray:
    device_buffers = []
    for device in jax.local_devices():
      local_indices = host_local_indices[device]
      data = element[local_indices]
      device_buffers.append(jax.device_put(data, device))
    return GlobalDeviceArray(shape, global_mesh, axes, device_buffers)

  pytree_of_gdas = jax.tree_map(
      form_gda, local_data, global_data_shape, data_axes)

  return pytree_of_gdas


################################################################################
### Shard data parallelism over devices, reshard inside pjit  (pax method) #####
################################################################################


# TODO(sholto): account for slicing / padding requirements
def get_fully_sharded_data_pipeline(
    dataset: tf.data.Dataset, global_data_shape: np.ndarray, global_mesh: Mesh,
    data_axes: P) -> Callable[[], GlobalDeviceArray]:
  """Test where each device loads batch_size/num_devices, then reshards in pjit.
  To do this, each host first loads batch_size/num_hosts, then shards that
  equally across it's devices.
  + Lowest data volume
  + Low complexity
  - Padding and slicing required when batch is not divisible by num_devices
  - Reshard takes time in pjit
  Args:
    dataset: tf dataset over all files
    global_data_shape: what the size of the GDA should be
    global_mesh: global deivces mesh
    data_axes: axes along which data is partitioned
  Returns:
    sharded_dataset: per_host dataset
  """
  batch = check_inputs(dataset, global_data_shape, data_axes)
  per_host = batch // jax.process_count()
  sharded_dataset = iter(
      dataset.shard(num_shards=jax.process_count(),
                    index=jax.process_index())
              .batch(per_host)
              .repeat()
              .as_numpy_iterator())

  next_fn = partial(get_next_fully_sharded, sharded_dataset, global_data_shape,
                    global_mesh, data_axes)

  return next_fn


def reshard_fn(input_constraints,
               input_gda: GlobalDeviceArray) -> GlobalDeviceArray:
  '''Infer sharding from shape.
    We need to do this because we need some way of looking up
    sharding for arbitrary pytrees inside pjit. Pax uses the
    rank - this ought to be better, but it isn't perfect!
    There must be a better way.
    '''
  # TODO(sholto): pax has initial reshapes to prevent unnecessary
  #               halo exchanges. Understand and implement.
  # TODO(sholto): in the step fn is also where we would remove padding
  return jax.tree_map(with_sharding_constraint, input_gda, input_constraints)



def get_next_fully_sharded(local_dataset: tf.data.Dataset,
                           global_data_shape: Pytree, global_mesh: Mesh,
                           data_axes: Pytree) -> GlobalDeviceArray:
  """Splits the host loaded data equally over all devices."""

  local_data = local_dataset.next()

  local_devices = jax.local_devices()
  local_device_count = jax.local_device_count()

  def _put_to_devices(x):
    try:
      per_device_arrays = np.split(x, local_device_count, axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f'Unable to put to devices shape {x.shape} with '
          f'local device count {local_device_count}') from array_split_error
    device_buffers = [
        jax.device_put(arr, d)
        for arr, d in zip(per_device_arrays, local_devices)
    ]
    return device_buffers

  # 'fully shard' the data (first) axis across both axes
  # of the hardware mesh. This is layout matches the
  # manual device placing we just did.
  input_sharding_constraint = P(('data', 'model'), None)

  def form_gda(local_data, shape):
    device_buffers = _put_to_devices(local_data)
    #  Wrap device buffers as GDA
    shape = tuple(shape)
    input_gda = GlobalDeviceArray(shape, global_mesh,
                                  input_sharding_constraint, device_buffers)
    return input_gda

  input_gdas = jax.tree_map(form_gda, local_data, global_data_shape)

  shape_to_sharding = {}
  # TODO(sholto): Find a functionally pure way to do this
  def update_shape_to_sharding(gda: GlobalDeviceArray, sharding: P):
    shape_to_sharding[jnp.shape(gda)] = sharding
  jax.tree_map(update_shape_to_sharding, input_gdas, data_axes)

  # Everything between here and the comment below should be inside your step fn
  # reshard inside pjit - involves sending data over ICI to correct devices
  # TODO(sholto):  return this instead of doing two pjit calls
  # at some point before you call pjit, you will have flattened
  # and extracted your inputs so fine to flatten in the correctness test
  # this needs a different testing path
  inputs_to_pjit, _ = jax.tree_util.tree_flatten(input_gdas)
  output_constraints, _ = jax.tree_util.tree_flatten(data_axes)

  inputs_to_pjit = tuple(inputs_to_pjit)
  output_constraints = tuple(output_constraints)

  # TODO(sholto): Fix this hack
  def matching_input_structure(no_op):
    return input_sharding_constraint
  input_constraints = jax.tree_map(matching_input_structure, output_constraints)

  reshard = pjit(
      partial(reshard_fn, output_constraints),
      in_axis_resources=(input_constraints,),
      out_axis_resources=output_constraints)

  with global_mesh:
    desired_gda = reshard(inputs_to_pjit)
  # Everything between here and the comment above should be inside your step fn
  # TODO(sholto): To make all methods comparable, maybe we should call a dummy
  #               pjit op for all after these get fns?

  return desired_gda


################################################################################
########################## Use tensorstore #####################################
################################################################################

# TODO(sholto): ^^

################################################################################
###################### Load on one, distribute over dcn ########################
################################################################################

# TODO(sholto): ^^

################################################################################
########################### Common code ########################################
################################################################################


def test_correctness(method: str):
  """Generic code to set up the tests of each pipeline ethod."""
  print(f'----------- Now testing {method} method ------------------------')
  # Initialise our desired GDA shape and device mesh layout
  batch = 32
  # Use "P" tuple class to treat global shapes as leaves:
  global_data_shapes = (P(batch, 4), P(batch, 2))
  data_axes = (P('data', None), P('data', None))
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

  # Create a test dataset

  global_data = tuple([
      np.arange(np.prod(shape)).reshape(shape) for shape in global_data_shapes
  ])
  dataset = tf.data.Dataset.from_tensor_slices(global_data)

  method_to_fn = {
      'all_data_all_hosts': get_all_data_all_hosts_pipeline,
      'per_replica': get_per_replica_data_pipeline,
      'per_host': get_per_host_data_pipeline,
      'fully_sharded': get_fully_sharded_data_pipeline,
  }

  next_batch_fn = method_to_fn[method](dataset, global_data_shapes, global_mesh,
                                       data_axes)
  gda = next_batch_fn()

  test = partial(test_gda_output, method, test_mesh_layout)
  jax.tree_map(test, global_data, gda)


################################################################################
########################### Test cases #########################################
################################################################################


def test_gda_output(method: str, test_mesh_layout: np.ndarray,
                    global_data: np.ndarray, gda: GlobalDeviceArray):
  """Compares against a known GDA arrangement - testmesh32."""
  print(f'First device: \n {gda.local_data(0)}')
  print(f'Fifth device: \n {gda.local_data(4)}')

  replicas = test_mesh_layout.shape[data_dim]
  half = global_data.shape[data_dim] // 2
  quarter = half // 2

  if method == 'per_replica':
    # Per host re-indexes s.t. there is only one tf.data shard per host
    if jax.process_index() == 0 or jax.process_index() == 1:
      assert (gda.local_data(0) == global_data[0::replicas]).all()
      assert (gda.local_data(4) == global_data[1::replicas]).all()
    if jax.process_index() == 2 or jax.process_index() == 3:
      assert (gda.local_data(0) == global_data[2::replicas]).all()
      assert (gda.local_data(4) == global_data[3::replicas]).all()

  elif method == 'per_host':
    # Per host re-indexes s.t. there is only one tf.data shard per host
    unique_shards = 2  # custom for the testmesh layout
    if jax.process_index() == 0 or jax.process_index() == 1:
      assert (gda.local_data(0) == global_data[0::unique_shards][:quarter]).all()  # pylint: disable=line-too-long
      assert (gda.local_data(4) == global_data[0::unique_shards][quarter:]).all()  # pylint: disable=line-too-long
    if jax.process_index() == 2 or jax.process_index() == 3:
      assert (gda.local_data(0) == global_data[1::unique_shards][:quarter]).all()  # pylint: disable=line-too-long
      assert (gda.local_data(4) == global_data[1::unique_shards][quarter:]).all()  # pylint: disable=line-too-long

  elif method == 'fully_sharded':
    # TODO(sholto): Test case for pax method
    print('Test case in progress as optimising the reshard may change the end layout. Please visually inspect.')  # pylint: disable=line-too-long

  else:

    if jax.process_index() == 0 or jax.process_index() == 1:
      assert (gda.local_data(0) == global_data[0:half][:quarter]).all()
      assert (gda.local_data(4) == global_data[0:half][quarter:]).all()
    if jax.process_index() == 2 or jax.process_index() == 3:
      assert (gda.local_data(0) == global_data[half:][:quarter]).all()
      assert (gda.local_data(4) == global_data[half:][quarter:]).all()

  print(f"Method: {method} '\u2713'")


if __name__ == '__main__':
  # test_correctness('all_data_all_hosts')
  # test_correctness('per_replica')
  # test_correctness('per_host')
  test_correctness('fully_sharded')

# Note: tests which set up one host to have multiple processes
#  https://source.corp.google.com/piper///depot/google3/learning/brain/research/jax/tests/tpu/multiprocess_tpu_test.py