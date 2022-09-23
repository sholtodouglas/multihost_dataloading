

Install deps

```
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install --user --upgrade -U "tensorboard>=2.3"
pip3 install --user --upgrade -U "tensorflow>=2.3"
pip3 install -U tensorboard-plugin-profile
pip3 install cloud-tpu-client
```

Ensure you are running a file with the following up top

```
tf.profiler.experimental.server.start(6000)
```

```
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central1-a --ssh-flag="-4 -L 9001:localhost:9001"
```
Run TensorBoard in the terminal window you just opened and specify the directory where TensorBoard can write profiling data with the --logdir flag. For example:

```
TPU_LOAD_LIBRARY=0 tensorboard --logdir your-model-dir --port 9001

 ```

 Go to the browser link

 ```
http://localhost:9001/
 ```