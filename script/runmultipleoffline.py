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

    # scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ~/Desktop/models.zip "+username+'@'+hosts[host_id]+":"+parent_dir+";"
    # print(scp_cmd)
    # os.system(scp_cmd)
    # unzip_cmd = "cd "+parent_dir+"; tmux new-session -d unzip models.zip"
    # stdin, stdout, stderr = rssh_object.exec_command(unzip_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)

    # setup node
    # setup_cmd = "ssh-keyscan "+hosts[host_id]+" >> $HOME/.ssh/known_hosts"
    # print(setup_cmd)
    # os.system(setup_cmd)
    clone_cmd = "pip3 install torch sklearn pynverse; rm -rf /mydata/output/z*;" # rm -rf /mydata/traces/zi*; rm -rf /mydata/traces/*.zip; sudo apt-get update; sudo apt-get install -y python3.6 libjpeg-dev zlib1g-dev; sudo apt-get install -y python3-pip; pip3 install numpy scipy PySide2 datetime matplotlib; ssh-keyscan github.com >> ~/.ssh/known_hosts; sudo chown -R janechen /mydata; pip3 install bloom-filter2; mkdir -p "+parent_dir+"traces; mkdir -p "+parent_dir+"output; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; chmod +x script/collect.sh; chmod +x script/collectsub.sh; chmod +x script/online.sh; chmod +x script/onlinesub.sh;"
    stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
    # scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /home/jane/Downloads/SyntheticTrace/trace"+str(host_id)+".zip "+username+'@'+hosts[host_id]+":"+parent_dir+"traces/;"
    # print(scp_cmd)
    # os.system(scp_cmd)
    # prepare_cmd = "tmux new-session -d unzip "+parent_dir+"traces/trace"+str(host_id)+".zip -d /mydata/traces"
    # stdin, stdout, stderr = rssh_object.exec_command(prepare_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)

def check(host_id):
    rssh_object = ssh_hosts[host_id]

    check_cmd = "cd ~/MultiExpertHOCAdmission; chmod +x ./script/checktraces.sh; ./script/checktraces.sh"
    print(check_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def check_results(host_id):
    rssh_object = ssh_hosts[host_id]

    check_cmd = "rm -rf /mydata/output/trace*; cd ~/MultiExpertHOCAdmission; git pull; python3 ./script/checkresults.py > results."+str(host_id)
    print(check_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/output/results.pkl"+" ~/Downloads/results/results"+str(host_id)+".pkl"
    print(scp_cmd)
    os.system(scp_cmd)

def run(host_id):
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "cd ~/MultiExpertHOCAdmission; git pull; tmux new-session -d ./script/collect.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
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
        check_results(i)
        # setup(i)
        # check(i)
        # run(i)