'''Stress tests the dataloaders using different video formats.

Useful for profiling.
'''

from jax.experimental import global_device_array as gda_lib
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import Device
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable CUDA not found etc warnings
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time

from dataloaders import construct_test_mesh_32, get_per_replica_data_pipeline

tf.profiler.experimental.server.start(6000)

# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True)

def stress_test(dataset):

  print('Begin stress test')
  global_data_shape = (512, 28, 28, 1)
  data_axes = P('data', None, None, None)
  # Create our device mesh - this function arranges a 4 hosts/32 devices
  # to allow us to test the general case
  test_mesh_layout = construct_test_mesh_32()
  global_mesh = Mesh(test_mesh_layout, ('data', 'model'))
  print('Mesh constructed')
  # imagine these values are host id, and the numbers are each a device
  # we create four replicas, each split over two hosts
  #     00001111
  #     00001111
  #     22223333
  #     22223333 
  
  next_batch_fn = get_per_replica_data_pipeline(dataset, global_data_shape, global_mesh,
                                       data_axes)

  def dummy_fn(x):
    return x + 1
  
  fn = pjit(dummy_fn, in_axis_resources=data_axes, out_axis_resources=data_axes)

  while(1):
    gda = next_batch_fn()
    with global_mesh:
      y = fn(gda)
      print(y)


if __name__ == "__main__":
  stress_test(ds)