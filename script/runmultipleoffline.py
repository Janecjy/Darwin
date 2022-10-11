import argparse
import os
import paramiko
import yaml
import math

hosts = []
username = ''
ssh_hosts = []
parent_dir = ''
trace_path = ''
output_path = ''

def connect_rhost(rhost, username):
    rssh = paramiko.client.SSHClient()
    # rssh.load_system_host_keys()
    rssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
    rssh.connect(hostname=rhost, username=username)
    s = rssh.get_transport().open_session()
    paramiko.agent.AgentRequestHandler(s)
    rssh.get_transport().set_keepalive(50000)
    return rssh

def setup(host_id):
    rssh_object = ssh_hosts[host_id]

    # setup node
    clone_cmd = "pip3 install bloom-filter2; rm /mydata/traces/debug.txt /mydata/traces/logfile.txt; rm -rf "+parent_dir+"output; mkdir -p "+parent_dir+"output; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; chmod +x script/collect.sh; chmod +x script/collectsub.sh;"
    stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)

def check(host_id):
    rssh_object = ssh_hosts[host_id]

    check_cmd = "cd ~/MultiExpertHOCAdmission; chmod +x ./script/checktraces.sh; ./script/checktraces.sh"
    print(check_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def check_results(host_id):
    rssh_object = ssh_hosts[host_id]

    check_cmd = "cd ~/MultiExpertHOCAdmission; python3 checkresults.py > results."+host_id
    print(check_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":~/MultiExpertHOCAdmission/results."+host_id+" ~/Downloads/offline-results/"
    print(scp_cmd)
    os.system(scp_cmd)

def run(host_id):
    rssh_object = ssh_hosts[host_id]
    run_cmd = "cd ~/MultiExpertHOCAdmission; tmux new-session -d ./script/collect.sh "+trace_path+" "+output_path
    print(run_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Offline run simulator experiments')
    parser.add_argument('-f', action="store", dest="config_file_path")
    args = parser.parse_args()
    config_file_path = args.config_file_path
    fp = open(config_file_path, "r")
    yaml_obj = yaml.safe_load(fp)
    hosts = yaml_obj['hosts']
    username = yaml_obj['username']
    parent_dir = yaml_obj['trace_parent_dir']
    trace_path = parent_dir+"traces/"
    output_path = parent_dir+"output/"
    
    print('total_hosts:', len(hosts))

    for i in range(0, len(hosts)):
        host = hosts[i]
        print(host)
        ssh_object = connect_rhost(host, username)
        ssh_hosts.append(ssh_object)
        setup(i)
        check(i)
        run(i)