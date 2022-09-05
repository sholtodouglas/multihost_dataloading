from ipaddress import ip_address
from os import system 
import argparse
import subprocess
import requests
import time
import os


parser = argparse.ArgumentParser(description='Setup slice workflow.')
parser.add_argument('--tpu_name')
parser.add_argument('--zone')
parser.add_argument('--project')
parser.add_argument("--skip_ssh_key", default=False, action="store_true",
                    help="skip the setup steps, e.g. if a later step fails")
parser.add_argument("--skip_ssh_connection", default=False, action="store_true",
                    help="skip the setup steps, e.g. if a later step fails")
parser.add_argument("--skip_setup_fswatch", default=False, action="store_true",
                    help="skip the setup steps, e.g. if a later step fails")
parser.add_argument("--working_dir", default='working_dir/',
                    help="dir to sync with fswatch")

def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()

def tpu_info(name, project, zone):
  headers = {'Authorization': f'Bearer {get_bearer()}'}

  response = requests.get(
    f"https://tpu.googleapis.com/v2alpha1/projects/{project}/locations/{zone}/nodes/{name}",
    headers=headers)

  return response.json()

def tmux(command):
  status = system('tmux %s' % command)
  return status

def tmux_shell(command):
  tmux('send-keys "%s" "C-m"' % command)

def tmux_select_pane(idx):
  tmux(f'select-pane -t {idx}')


if __name__ == "__main__":
  args = parser.parse_args()

  print(args.tpu_name)

  # Attach to the window titled pod_control_pane in the background
  tmux("attach -t -d pod_control_pane")
  
  # Get information about the tpu - specifically the # of hosts and their IP addresses
  info = tpu_info(args.tpu_name, args.project, args.zone)

  num_hosts = len(info['networkEndpoints'])
  print(f"Total hosts: {num_hosts}")

  for idx, addr in enumerate(info['networkEndpoints']):
    print(f"Host {idx}: Internal IP: {addr['ipAddress']} External IP {addr['accessConfig']['externalIp']}")

  internal_ips = [addr['ipAddress'] for addr in info['networkEndpoints']]
    #############################################################################
  ########## SSH key config so that the main host can talk to the others ######
  #############################################################################

  # create a key
  if args.skip_ssh_key:
    print("Skipping creating intra-key hosts - you indicated this is already done.")
  else:
    p = subprocess.Popen("ssh-keygen -t rsa -f ~/.ssh/pod_key -N '' -C 'my_tpu_pod'", shell=True, stdin=subprocess.PIPE)
    p.stdin.write(b"y")
    outputlog, errorlog = p.communicate()
    p.stdin.close()

    # Copy file over to each TPU for setup
    try:
      # Make the key we created authorised for all TPUs
      output = subprocess.run(f'gcloud compute tpus tpu-vm scp ~/.ssh/pod_key.pub {args.tpu_name}:.ssh/pod_key.pub --worker=all --zone={args.zone}', shell=True, check=True)
      # give all workers the private key - we could only give worker 0: TODO: Check security implications
      output = subprocess.run(f'gcloud compute tpus tpu-vm scp ~/.ssh/pod_key {args.tpu_name}:.ssh/pod_key --worker=all --zone={args.zone}', shell=True, check=True)

    except subprocess.CalledProcessError as error:
      print("Note - if this threw an error, as indicated it will likely be the fact that you haven't used scp before. Follow the fix provided in the gcp output, then rerun this script with --skip_ssh_connection to skip the ssh setup steps.")


  #############################################################################
  ########## Create a window per host & ssh directly into each one ############
  #############################################################################
  if args.skip_ssh_connection:
    print("Skipping connecting to hosts - you indicated you already have separate tmux windows with each host connected.")
  else:
    for i in range(0, num_hosts-1):
      tmux("split-window -h") # TODO
      
    # Layout the windows nicely
    tmux("select-layout tiled")

    # If you get stuck connecting here - make sure you can connect normally. 
    # E.g. if you are a googler, run gcert. 
    for i in range(0, num_hosts):
      tmux_select_pane(i)  
      tmux_shell(f"gcloud compute tpus tpu-vm ssh {args.tpu_name} --zone {args.zone} --project {args.project} --worker {i}")

    tmux_select_pane(0)  
    # terminal broadcast across all panes
    tmux("setw synchronize-panes")


    # add this key to the approved hosts for each 
    tmux_shell('cat .ssh/pod_key.pub >> .ssh/authorized_keys')
    # and correct the permissions
    tmux_shell('chmod 600 ~/.ssh/pod_key')
  
      # add all internal ips to eachother's known hosts, so that rsync works off the bat (avoids needing to unset strictHostChecking)
    tmux_shell(f"ssh-keyscan -H {' '.join(internal_ips)} >> ~/.ssh/known_hosts")
    # If you wanted, you could now go to the tmux terminal of any of the machines, turn off broadcasting and then ssh into any other machine using
    #  'ssh -i ~/.ssh/pod_key.pub INSERT_INTERNAL_IP_HERE', where the internal ip addresses were printed earlier by our script (or you can look them up in gcp). 
    # This is what we will use to automatically synchronize files from host 0 to the others.

    # modify this to change the watched working dir
    tmux_shell(f'mkdir {args.working_dir}')
    tmux_shell(f'cd {args.working_dir}')

  # #############################################################################
  # ########## Set up the fswatch sync across working dirs ############
  # #############################################################################
  if args.skip_setup_fswatch:
    print("Skipping fswatch - you're already syncing")
  else:
  # create a new window, where we will launch fswatch from
    tmux("new-window")
    # connect to our main worker
    tmux_shell(f"gcloud compute tpus tpu-vm ssh {args.tpu_name} --zone {args.zone} --project {args.project} --worker {0}")

    tmux_shell('sudo apt install fswatch')

    # create the rsync file call sync.sh
    working_destinations = ' '.join([ip_address+':'+args.working_dir for ip_address in internal_ips[1:]])
    ssh_access = "'ssh -i ~/.ssh/pod_key'"
    file_create_string = f'echo "for d in {working_destinations}; do rsync -a -e {ssh_access} {args.working_dir} \$d; done" > sync.sh'
  
    tmux_shell(file_create_string)
    tmux_shell('chmod +x sync.sh')

    # sync once every 3 seconds
    tmux_shell(f'fswatch --one-per-batch --recursive --latency 3 --verbose {args.working_dir} | xargs -I{{}} ./sync.sh')

    # go back
    tmux("select-window -t 0")

  #############################################################################
  ########## Run setup scripts on each TPU ############
  #############################################################################
  

  # Copy file over to each TPU for setup
  # try:
  #   output = subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", "scp", "example.py", f"{args.tpu_name}:", "--worker=all", f"--zone={args.zone}" ], check=True)
  #   print(output)
  # except subprocess.CalledProcessError as error:
  #   print("Note - if this threw an error, as indicated it will likely be the fact that you haven't used scp before. Follow the fix provided in the gcp output, then rerun this script with --skip_ssh_connection to skip the ssh setup steps.")

  # for i in range(0, num_hosts):
  #     tmux_select_pane(i)  
  #     tmux_shell(f"python3 example.py")

  # gcloud compute tpus tpu-vm scp ~/.ssh/pod_key.pub sholto-v2-32: --worker=all --zone=us-central1-a