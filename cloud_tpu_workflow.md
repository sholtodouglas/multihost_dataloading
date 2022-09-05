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


```
# Go to a new terminal window, and create this tmux session
tmux new-session -d -s pod_control_pane

# From your original terminal window, run this python script which will connect to the tmux session, create a window for every host in the TPU slice, setup terminal broadcasting. 
python3 workflow_setup/infra.py --tpu_name=$TPU_NAME --zone=$ZONE --project=$PROJECT
```
 
Now, we should have two terminals. 
- Local: The terminal one your local machine you just ran workflow_setup/infra.py from
- Remote: A tmux terminal connected to every host, with terminal broadcasting.

In your remote terminal
