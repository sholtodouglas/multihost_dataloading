Create our system variables

```
TPU_NAME=sholto-v2-32
ZONE=us-central1-a
PROJECT=jax-dev
```

Create the pod slice
```
gcloud compute tpus tpu-vm create $TPU_NAME   --zone $ZONE   --project $PROJECT --accelerator-type v2-32   --version tpu-vm-base
```

To interact with the pod slice, we need to run a python file in parallel on all of them + do assorted config. 

A simple but manual process is described [here](https://cloud.google.com/tpu/docs/jax-pods). 

We're expecting to have a more interactive dev process, so we'll trade off a little setup for a quicker repeated workflow. We'll run a script that sets up a tmux pane, for each host in the slice - and sets up terminal broadcasting so that typing in any pane will be copied to the others. 

If you have never used scp before

```
ssh-add ~/.ssh/google_compute_engine
```

```
# Go to a new terminal window, and create this tmux session
tmux new-session -s pod_control_pane

# From your original terminal window, run this python script which will connect to the tmux session, create a window for every host in the TPU slice, setup terminal broadcasting. 
python3 workflow_setup/infra.py --tpu_name=$TPU_NAME --zone=$ZONE --project=$PROJECT
```
 
Now, we should have two terminals. 
- Local: The terminal one your local machine you just ran workflow_setup/infra.py from
- Remote: A tmux terminal connected to every host, with terminal broadcasting.

NOTE: If you want to turn off broadcasting to work in only one window, type ctrl-b then : to open a prompt. Then type 'setw synchronize-panes'. This toggles it on and off. 

These next steps we are doing outside a setup script for the moment - TODO: wrap them up. 
In your local terminal

```
ssh-keygen -t rsa -f ~/.ssh/pod_key -N '' -C 'my_tpu_pod'
# give all workers the public key so we can copy it to authorised hosts
gcloud compute tpus tpu-vm scp ~/.ssh/pod_key.pub $TPU_NAME:.ssh/pod_key.pub --worker=all --zone=$ZONE
# give all workers the private key - we could only give worker 0: TODO: Check security implications
# this allows any machine to ssh to any other machine
gcloud compute tpus tpu-vm scp ~/.ssh/pod_key $TPU_NAME:.ssh/pod_key --worker=all --zone=$ZONE

```
In your remote terminal

```
# include the pubkey in authorised keys
cat .ssh/pod_key.pub >> .ssh/authorized_keys
chmod 600 ~/.ssh/pod_key

```

If you wanted, you could now go to the tmux terminal of any of the machines, turn off broadcasting and then ssh into any other machine using 'ssh -i ~/.ssh/pod_key.pub INSERT_INTERNAL_IP_HERE', where the internal ip addresses were printed earlier by our script (or you can look them up in gcp).
