from os import system
import argparse
import subprocess, signal
import os


parser = argparse.ArgumentParser(description='Kill hanging multihost programs.')
parser.add_argument('--tpu_name')
parser.add_argument('--zone')
parser.add_argument('--project')
parser.add_argument('--proc_name')


if __name__ == "__main__":
  args = parser.parse_args()
  command = f"pkill -f {args.proc_name}"
  output = subprocess.run(f'gcloud compute tpus tpu-vm ssh {args.tpu_name} --zone {args.zone} --project {args.project} --worker=all --command "{command}"', shell=True)

print('If you see command execution failed.., it likely means there was not process of that name running on that host')