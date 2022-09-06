We'll be working with two terminal windows here - one locally for setup, and one connected to our host TPUs in the slice.

### Go to a one terminal window to create the tmux session we'll be using for remote access
```yaml
# Go to a new terminal window, and create this tmux session
tmux new-session -s pod_control_pane
```

### On another terminal window

Create our system variables

```javascript
TPU_NAME=sholto-v2-32
ZONE=us-central1-a
PROJECT=jax-dev
```

Create the pod slice
```ruby
gcloud compute tpus tpu-vm create $TPU_NAME   --zone $ZONE   --project $PROJECT --accelerator-type v2-32   --version tpu-vm-base
```

To interact with the pod slice, we need to run a python file in parallel on all of them + do assorted config. 

A simple but manual process is described [here](https://cloud.google.com/tpu/docs/jax-pods). 

We're expecting to have a more interactive dev process, so we'll trade off a little setup for a quicker repeated workflow. We'll run a script that sets up a tmux pane, for each host in the slice - and sets up terminal broadcasting so that typing in any pane will be copied to the others. In the background, it'll also setup fswatch + rsync so that files are synced from the host to the others every 3 seconds.

If you have never used scp with gcp before, you should do this

```ruby
ssh-add ~/.ssh/google_compute_engine
```

```ruby
# From your original terminal window, run this python script which will connect to the tmux session, create a window for every host in the TPU slice, setup terminal broadcasting. 
python3 workflow_setup/setup_hosts.py --tpu_name=$TPU_NAME --zone=$ZONE --project=$PROJECT
```
 
Now, we should have two terminals. 
- Local: The terminal one your local machine you just ran workflow_setup/infra.py from
- Remote: A tmux terminal connected to every host, with terminal broadcasting. In the background (on window 1) should be a process running fswatch, syncing anything you do on host 1 with all the others.

> **Note**
> If you want to turn off broadcasting to work in only one window, type ctrl-b then : to open a prompt. Then type 'setw synchronize-panes'. This toggles it on and off. 

> **Note**
> If you wanted, you could now go to the tmux terminal of any of the machines, turn off broadcasting and then ssh into any other machine using 'ssh -i ~/.ssh/pod_key INSERT_INTERNAL_IP_HERE', where the internal ip addresses were printed earlier by our script (or you can look them up in gcp). This is what we will use to automatically synchronize files from host 0 to the others.

### Editing code

Now you can edit code anyway you like!

Personally, I like to use vs-code remote to jump directly into host 0 and make edits there. Then, when you want to run a file take advantage of the terminal broadcasting to simply run it across all panes in tmux as though you had one terminal. Any changes you make will be reflected in the background by fswatch + rsync.

Other options I've heard that work well are emacs on host 0, using terminal broadcasting!

> **Warning**
> If the files have not synced and you run them, they may hang. One way to confirm they've synced is to run 'md5sum FILENAME' and confirm the hashes are the same. 
> Anything in 'working_dir' will be replicated across tpus - so aim not to store weights or data in there. 

> **Note**
> If you interrupt the script during scp, gcp is sometimes interrupted while writing to google_compute_known_hosts, and will fail the next time you try as the keys of the remote address do not match what it expects. To fix this, go to ~/.ssh/google_compute_known_hosts, and delete the lines corresponding to this tpu so it can start fresh.

