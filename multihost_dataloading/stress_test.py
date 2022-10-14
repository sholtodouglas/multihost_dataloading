'''Stress tests the dataloaders using different video formats.

Useful for profiling.
'''

import jax
import jax.numpy as jnp
from jax.experimental import global_device_array as gda_lib
from jax.experimental import PartitionSpec as P
from jax.experimental.global_device_array import Device
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
import os
from multihost_dataloading.dataloaders import get_per_host_data_pipeline
from multihost_dataloading.dataloaders import get_all_data_all_hosts_pipeline
from multihost_dataloading.dataloaders import get_fully_sharded_data_pipeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable CUDA not found etc warnings
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time


from dataloaders import construct_test_mesh_32, get_per_replica_data_pipeline
import dataloaders

NUM_STEPS = 15

print('---------------- * ---------------')

def create_fake_image_dataset():
  b, h, w, f, c = 32, 224, 224, 32, 3
  l = 1000
  np.random.seed(42)
  fake_imgs = np.random.uniform(size = (b, h, w, f, c))
  classes =  np.random.uniform(size= (b, l))
  dataset = tf.data.Dataset.from_tensor_slices({"images": fake_imgs, "classes": classes}).repeat()
  global_data_shapes = {"images": P(b, h, w, f, c), "classes": P(b, l)}
  data_axes = {"images": P('data', None, None, None, None), "classes": P("data", None)}
  return dataset, global_data_shapes, data_axes

# def load_imagenet64():
#   dataset = tfds.load('imagenet_resized/64x64', split='train', data_dir='gs://tensorflow-datasets/datasets').prefetch(256)
#   # print(jax.tree_map(jnp.shape, iter(dataset).next()))
#   print('dataset created')
#   b, h, w, c = 8, 64, 64, 3
#   global_data_shapes = {'image': P(b, h, w, c), 'label': P(b)}
#   data_axes = {"image": P('data', None, None, None), "label": P("data")}
#   return dataset, global_data_shapes, data_axes

def load_imagenet64():
  dataset = tfds.load('imagenet_resized/64x64', split='train', data_dir='~/tensorflow-datasets/datasets')
  # print(jax.tree_map(jnp.shape, iter(dataset).next()))
  print('dataset created')
  b, h, w, c = 8, 64, 64, 3
  global_data_shapes = {'image': P(b, h, w, c), 'label': P(b)}
  data_axes = {"image": P('data', None, None, None), "label": P("data")}
  return dataset, global_data_shapes, data_axes

def load_mnist():
  dataset = tfds.load('mnist', split='train', data_dir='gs://tensorflow-datasets/datasets')
  # print(jax.tree_map(jnp.shape, iter(dataset).next()))
  print('dataset created')
  b, h, w, c = 8, 28, 28, 1
  global_data_shapes = {'image': P(b, h, w, c), 'label': P(b)}
  data_axes = {"image": P('data', None, None, None), "label": P("data")}
  return dataset, global_data_shapes, data_axes

def load_youtube():
  yt_ds = tfds.load('youtube_vis/480_640_full', split='train', data_dir='gs://multihost_datadir_central_1a/tensorflow-datasets/datasets')
  yt_ds = yt_ds.map(lambda x: x['video'][:10, ...])
  b, t, h, w, c = 32, 10, 480, 640, 3
  global_data_shapes =  P(b, t, h, w, c)
  data_axes = P('data', None, None, None, None)
  return yt_ds, global_data_shapes, data_axes


def fake_flops(inputs):
  x  = inputs
  x  = jnp.einsum('bthwc,bthwc->thc', x, x)
  return x


def stress_test(dataset, global_data_shape, data_axes):

  print('---------------- Test ---------------')
  print(global_data_shape)        
  
  # Create our device mesh - this function arranges a 4 hosts/32 devices
  # to allow us to test the general case
  test_mesh_layout = construct_test_mesh_32()
  global_mesh = Mesh(test_mesh_layout, ('data', 'model'))
  # print('Mesh constructed')
  # imagine these values are host id, and the numbers are each a device
  # we create four replicas, each split over two hosts
  #     00001111
  #     00001111
  #     22223333
  #     22223333 
  
  next_batch_fn = dataloaders.get_per_host_data_pipeline(dataset, global_data_shape, global_mesh,
                                       data_axes)
  
  output_partitioning = P(None, None, None)
  fn = pjit(fake_flops, in_axis_resources=(data_axes,), out_axis_resources=output_partitioning)
  
  for i in range(0,NUM_STEPS):
    if i == 1:
      t1 = time.time()

    t1_1 = time.time()
    gda = next_batch_fn()
    t1_2 = time.time()
    
    # print(jax.tree_map(jnp.shape, gda))
    with global_mesh:
      y = fn(gda).block_until_ready()
      print(t1_2-t1_1, y.shape)

  t2 = time.time()
  print(f"{(t2-t1)/NUM_STEPS}s per step")

dataset = 'youtube'

if __name__ == "__main__":
  if dataset == 'youtube':
    yt_ds, shapes, axes = load_youtube()
    stress_test(yt_ds, shapes, axes)
  elif dataset == 'iamgenet64':
    im64_dataset, shapes, axes  = load_imagenet64()
    stress_test(im64_dataset, shapes, axes)
  else:
    imgnet_dataset, imgnet_shapes, data_axes = create_fake_image_dataset()
    stress_test(imgnet_dataset, imgnet_shapes, data_axes)