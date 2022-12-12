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
    clone_cmd = "mkdir -p /mydata/features-req" # "mkdir -p /mydata/traces-more; cd /mydata/traces; mv *.zip /mydata/"
    # stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    
    # for i in range(10):
    #     for j in range(10):
    #         clone_cmd = "cd /mydata/traces; mv *-"+str(i)+"-"+str(j)+".txt /mydata/traces-more/"
    #         stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    #         for line in iter(stdout.readline, ""):
    #             print(line)
                
    # for i in range(10):
    #     for j in range(10):
    #         clone_cmd = "cd /mydata/features; rm -rf ./*-"+str(i)+"-"+str(j) # rsync -rvh --remove-source-files ./*-"+str(i)+"-"+str(j)+" /mydata/features-more/"
    #         stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    #         for line in iter(stdout.readline, ""):
    #             print(line)

    # # setup node
    # if host_id >= 9:
    #     clone_cmd = "sudo apt-get update; sudo apt-get install -y python3.6 libjpeg-dev zlib1g-dev; sudo apt-get install -y python3-pip; ssh-keyscan github.com >> ~/.ssh/known_hosts; sudo apt-get install -y python3-venv unzip; rm -rf "+parent_dir+"features; mkdir -p "+parent_dir+"features; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; chmod +x script/collectfeature.sh; chmod +x script/collectfeaturesub.sh; python3 -m venv venv; source venv/bin/activate; pip install wheel numpy matplotlib bloom-filter2;"
    #     stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    #     for line in iter(stdout.readline, ""):
    #         print(line)
        
    # if host_id >= 14:
    #     clone_cmd = "cd /mydata/traces; tmux new-session -d unzip trace"+str(host_id)+".zip"
    #     stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    #     for line in iter(stdout.readline, ""):
    #         print(line)
    

def run(host_id):
    # if host_id >= 10 and host_id <= 14:
    rssh_object = ssh_hosts[host_id]
    run_cmd = "cd ~/MultiExpertHOCAdmission; git pull; tmux new-session -d ./script/collectfeature.sh "+trace_path+" "+output_path
    print(run_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def download(host_id):
    rssh_object = ssh_hosts[host_id]
    # run_cmd = "cd ~/MultiExpertHOCAdmission; git reset --hard; git pull; chmod +x ./script/calfeaturediff.sh; tmux new-session -d ./script/calfeaturediff.sh "+output_path
    # print(run_cmd)
    # stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    
    scp_cmd = 'scp -r '+username+'@'+hosts[i]+':/mydata/featurediff-req/* /Users/janechen/Documents/wisr/learned-cache/scripts/feature-results-req/'
    print(scp_cmd)
    os.system(scp_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Offline collect trace features')
    parser.add_argument('-f', action="store", dest="config_file_path")
    args = parser.parse_args()
    config_file_path = args.config_file_path
    fp = open(config_file_path, "r")
    yaml_obj = yaml.safe_load(fp)
    hosts = yaml_obj['hosts']
    username = yaml_obj['username']
    parent_dir = yaml_obj['trace_parent_dir']
    trace_path = parent_dir+"traces/"
    output_path = parent_dir+"features-req/"
    
    print('total_hosts:', len(hosts))

    for i in range(0, len(hosts)):
        host = hosts[i]
        print(host)
        ssh_object = connect_rhost(host, username)
        ssh_hosts.append(ssh_object)
        # setup(i)
        # run(i)
        download(i)