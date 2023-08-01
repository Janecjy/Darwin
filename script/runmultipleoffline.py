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

# def setup(host_id):
#     rssh_object = ssh_hosts[host_id]

#     # scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ~/Desktop/models.zip "+username+'@'+hosts[host_id]+":"+parent_dir+";"
#     # print(scp_cmd)
#     # os.system(scp_cmd)
#     # unzip_cmd = "cd "+parent_dir+"; tmux new-session -d unzip models.zip"
#     # stdin, stdout, stderr = rssh_object.exec_command(unzip_cmd, get_pty=True)
#     # for line in iter(stdout.readline, ""):
#     #     print(line)
        
#     rm_cmd = "rm -rf /mydata/experts/zi*"
#     stdin, stdout, stderr = rssh_object.exec_command(rm_cmd, get_pty=True)
#     for line in iter(stdout.readline, ""):
#         print(line)

    # setup node
    # setup_cmd = "ssh-keyscan "+hosts[host_id]+" >> $HOME/.ssh/known_hosts"
    # print(setup_cmd)
    # os.system(setup_cmd)
    # clone_cmd = "sudo apt-get install -y python3.6 libjpeg-dev zlib1g-dev; sudo apt-get install -y python3-pip; pip3 install numpy scipy PySide2 datetime matplotlib bloom-filter2; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; "
    # clone_cmd = "tmux kill-server; cd ~/MultiExpertHOCAdmission; git reset --hard; git checkout main; git pull; chmod +x script/distributedata.sh; chmod +x script/zipdistributeddata.sh; tmux new-session -d ./script/zipdistributeddata.sh"#; tmux new-session -d ./script/collect.sh /mydata/traces/ /mydata/output-offline/ 1 30" # torch sklearn pynverse; rm -rf /mydata/traces/zi*; rm -rf /mydata/traces/*.zip; sudo apt-get update; sudo apt-get install -y python3.6 libjpeg-dev zlib1g-dev; sudo apt-get install -y python3-pip; pip3 install numpy scipy PySide2 datetime matplotlib; ssh-keyscan github.com >> ~/.ssh/known_hosts; sudo chown -R janechen /mydata; pip3 install bloom-filter2; mkdir -p "+parent_dir+"traces; mkdir -p "+parent_dir+"output; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; chmod +x script/collect.sh; chmod +x script/collectsub.sh; chmod +x script/online.sh; chmod +x script/onlinesub.sh;"
    # # clone_cmd = "pip3 install bloom-filter2"
    # stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    # scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /home/jane/Downloads/SyntheticTrace/trace"+str(host_id)+".zip "+username+'@'+hosts[host_id]+":"+parent_dir+"traces/;"
    # print(scp_cmd)
    # os.system(scp_cmd)
    # prepare_cmd = "tmux new-session -d unzip "+parent_dir+"traces/trace"+str(host_id)+".zip -d /mydata/traces"
    # stdin, stdout, stderr = rssh_object.exec_command(prepare_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    
def setup(host_id):
    rssh_object = ssh_hosts[host_id]
    clone_cmd = "sudo apt update; sudo apt-get install -y python3.6 libjpeg-dev zlib1g-dev; sudo apt-get install -y python3-pip; pip3 install numpy scipy PySide2 datetime matplotlib bloom-filter2 scikit-learn scipy matplotlib; ssh-keyscan github.com >> ~/.ssh/known_hosts; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git "#chmod +x script/collect.sh; chmod +x script/collectsub.sh; chmod +x script/online.sh; chmod +x script/onlinesub.sh; torch"
    # clone_cmd = "sudo chown -R janechen /mydata;"# sudo apt-get install -y python3-pip; pip3 install numpy scipy bloom-filter2; ssh-keyscan github.com >> ~/.ssh/known_hosts; rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; git checkout windowbased-features; chmod +x script/percentile.sh;"
    # clone_cmd = "pip3 install numpy scipy PySide2 datetime matplotlib bloom-filter2 torch scikit-learn scipy matplotlib; ssh-keyscan github.com >> ~/.ssh/known_hosts; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission;"
    # clone_cmd = "ssh-keyscan github.com >> ~/.ssh/known_hosts; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd MultiExpertHOCAdmission; git checkout main; git reset --hard; git pull;"
    stdin, stdout, stderr = rssh_object.exec_command(clone_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def replicate(host_id):
    scp_cmd = "scp -3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/node1/node1.zip janechen@c220g5-111306.wisc.cloudlab.us:/mydata/"+str(host_id)+"-node1.zip &"
    print(scp_cmd)
    os.system(scp_cmd)
    scp_cmd = "scp -3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/node2/node2.zip janechen@c220g5-111306.wisc.cloudlab.us:/mydata/"+str(host_id)+"-node2.zip &"
    print(scp_cmd)
    os.system(scp_cmd)
    scp_cmd = "scp -3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/node3/node3.zip janechen@c220g5-111306.wisc.cloudlab.us:/mydata/"+str(host_id)+"-node3.zip &"
    print(scp_cmd)
    os.system(scp_cmd)
    scp_cmd = "scp -3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/node4/node4.zip janechen@c220g5-111306.wisc.cloudlab.us:/mydata/"+str(host_id)+"-node4.zip &"
    print(scp_cmd)
    os.system(scp_cmd)
    scp_cmd = "scp -3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/node5/node5.zip janechen@c220g5-111306.wisc.cloudlab.us:/mydata/"+str(host_id)+"-node5.zip &"
    print(scp_cmd)
    os.system(scp_cmd)

def run_online(host_id):
    # if host_id in [6, 10, 13]:
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "cd ~/MultiExpertHOCAdmission; git pull; tmux new-session -d ./script/online.sh "+trace_path+" "+output_path # +"; tmux new-session -d ./script/online3.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def gen_correlation(host_id):
    rssh_object = ssh_hosts[host_id]
    if hosts[host_id].endswith("clemson.cloudlab.us"):
        p_num = 40
    if hosts[host_id].endswith("wisc.cloudlab.us"):
        p_num = 32
    if hosts[host_id].endswith("apt.emulab.net"):
        p_num = 16
    if hosts[host_id].startswith("amd"):
        p_num = 32
    if hosts[host_id].startswith("hp"):
        p_num = 20
    run_cmd = "cd ~/MultiExpertHOCAdmission; git checkout main; git reset --hard; git pull; chmod +x ./script/gencorrelation.sh; tmux new-session -d ./script/gencorrelation.sh "+trace_path+" "+str(p_num) # +"; tmux new-session -d ./script/online3.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
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

    check_cmd = "rm -rf /mydata/output/results*; cd ~/MultiExpertHOCAdmission; git pull; python3 ./script/checkresults.py > results."+str(host_id)
    print(check_cmd)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/output-percentile/results.pkl"+" ~/Downloads/results-percentile/results"+str(host_id)+".pkl"
    print(scp_cmd)
    os.system(scp_cmd)
    
def check_online_percentile_results(host_id):
    rssh_object = ssh_hosts[host_id]

    # check_cmd = "rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; python3 ./script/checkresults-online.py; python3 ./script/checkresults-percentile.py"
    # print(check_cmd)
    # stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/output-percentile/results.pkl"+" ~/Downloads/darwin-data/results-percentile-new/results"+str(host_id)+".pkl"
    print(scp_cmd)
    os.system(scp_cmd)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/output-online/results.pkl"+" ~/Downloads/darwin-data/results-online-new/results"+str(host_id)+".pkl"
    print(scp_cmd)
    os.system(scp_cmd)

def check_hillclimbing_results(host_id):
    rssh_object = ssh_hosts[host_id]

    # check_cmd = "rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; cd ~/MultiExpertHOCAdmission; git pull; python3 ./script/checkresults-hillclimbing-con.py"
    # print(check_cmd)
    # stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    # for line in iter(stdout.readline, ""):
    #     print(line)
    scp_cmd = "scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+'@'+hosts[host_id]+":/mydata/output-hillclimbing-con/results.pkl"+" ~/Downloads/darwin-data/results-hillclimbing-con/results"+str(host_id)+".pkl"
    print(scp_cmd)
    os.system(scp_cmd)

def run(host_id):
    # if host_id in [6, 13]:
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; mkdir -p "+output_path+"; cd ~/MultiExpertHOCAdmission; git pull; chmod +x ./script/hillclimbing.sh; tmux new-session -d ./script/hillclimbing.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def run_percentile(host_id):
    # if host_id in [6, 13]:
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; mkdir -p "+output_path+"; cd ~/MultiExpertHOCAdmission; git pull; chmod +x ./script/percentile.sh; tmux new-session -d ./script/percentile.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def run_online(host_id):
    # if host_id in [6, 13]:
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; mkdir -p "+output_path+"; cd ~/MultiExpertHOCAdmission; git pull; chmod +x ./script/online.sh; chmod +x ./script/onlinesub.sh; tmux new-session -d ./script/online.sh "+trace_path+" "+output_path #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)
        
def run_percentile_online(host_id):
    rssh_object = ssh_hosts[host_id]
    print(output_path)
    run_cmd = "sudo pkill python; tmux kill-session; pip3 install pynverse; rm -rf ~/MultiExpertHOCAdmission; git clone git@github.com:Janecjy/MultiExpertHOCAdmission.git; mkdir -p "+output_path_online+" "+output_path_percentile+"; cd ~/MultiExpertHOCAdmission; git pull; chmod +x ./script/online.sh; chmod +x ./script/onlinesub.sh; chmod +x ./script/percentile.sh; tmux new-session -d ./script/online.sh "+trace_path+" "+output_path_online+"; tmux new-session -d ./script/percentile.sh "+trace_path+" "+output_path_percentile #+" ; ; "  tmux new-session -d ./script/online.sh "+trace_path+" "+output_path
    print(run_cmd)  
    stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)

def collect(host_id):
    if host_id in [6, 13]:
        rssh_object = ssh_hosts[host_id]
        run_cmd = "mkdir -p /mydata/experts; cd ~/MultiExpertHOCAdmission; git reset --hard; git pull; chmod +x ./script/collectoffline.sh;"
        print(run_cmd)  
        stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line)
        for f in [2, 3, 4, 5, 6, 7]:
            for s in [10, 20, 50, 100, 500, 1000]:
            # for s in [10, 20]:
                expert='f'+str(f)+'s'+str(s)
                run_cmd = "cd ~/MultiExpertHOCAdmission; tmux new-session -d ./script/collectoffline.sh "+expert+" /mydata/output/ /mydata/experts/"
                print(run_cmd)  
                stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
                for line in iter(stdout.readline, ""):
                    print(line)
                
def compress(host_id):
    if host_id == 10:
        for f in [2, 3, 4, 5, 6, 7]:
            for s in [10, 20, 50, 100, 500, 1000]:
            # for s in [10, 20]:
                expert='f'+str(f)+'s'+str(s)
                rssh_object = ssh_hosts[host_id]
                run_cmd = "cd /mydata/experts; rm -rf *.zip; tmux new-session -d zip -r "+expert+"-"+str(host_id)+".zip "+expert+"/*"
                print(run_cmd)  
                stdin, stdout, stderr = rssh_object.exec_command(run_cmd, get_pty=True)
                for line in iter(stdout.readline, ""):
                    print(line)

def transfer(host_id):
    if host_id == 6 or host_id == 13:
        rssh_object = ssh_hosts[host_id]
        # for f in [2, 3, 4, 5, 6, 7]:
        for f in [2, 3, 4]:
            for s in [10, 20]:
                expert = 'f'+str(f)+'s'+str(s)
                scp_command = "scp -3 "+username+'@'+hosts[host_id]+":/mydata/experts/"+expert+"/* janechen@c240g1-031313.wisc.cloudlab.us:/mydata/experts/"+expert+"/"
                os.system(scp_command)
                
def check_zip(host_id):
    rssh_object = ssh_hosts[host_id]
    # check_cmd = "ls /mydata/ | grep .zip"
    # check_cmd = "ls /mydata/correlations/f7s100-f7s10 | grep pkl | wc -l"
    check_cmd = "cd ~/MultiExpertHOCAdmission; git reset --hard; git pull; chmod +x ./script/zipcorrelation.sh; tmux new-session -d ./script/zipcorrelation.sh "+str(host_id)
    stdin, stdout, stderr = rssh_object.exec_command(check_cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Offline run simulator experiments')
    # parser.add_argument('-f', action="store", dest="config_file_path")
    # args = parser.parse_args()
    # config_file_path = args.config_file_path
    # fp = open(config_file_path, "r")
    # yaml_obj = yaml.safe_load(fp)
    # hosts = yaml_obj['hosts']
    # username = yaml_obj['username']
    # parent_dir = yaml_obj['trace_parent_dir']
    # trace_path = parent_dir+"traces/"
    # output_path = parent_dir+"output-offline/"
    # # trace_path = parent_dir+"traces/"
    # # output_path_percentile = parent_dir+"output-percentile/"
    # # output_path_online = parent_dir+"output-online/"
    
    # print('total_hosts:', len(hosts))

    # for i in range(0, len(hosts)):
    #     host = hosts[i]
    #     print(host)
    #     if i == 40:
    #         ssh_object = connect_rhost(host, username)
    #         ssh_hosts.append(ssh_object)
    #         setup(i)
    #         gen_correlation(i)
    #     else:
    #         ssh_hosts.append(1)
        # run_percentile(i)
        # if i >= 17:
        # run_percentile_online(i)
        # check_hillclimbing_results(i)
        # if i > 4:
        # replicate(i)
        # run_online(i)
        # check_results(i)
        # setup(i)
        # check(i)
        # if i == 1:
            # run(i)
        # collect(i)
        # compress(i)
        # transfer(i)
        # check_online_percentile_results(i)
        
    inactive_nodes = ["c220g1-030822.wisc.cloudlab.us", "c220g1-030826.wisc.cloudlab.us", "c220g1-030823.wisc.cloudlab.us", "apt115.apt.emulab.net"]
    parser = argparse.ArgumentParser(
        description='Offline run simulator experiments')
    parser.add_argument('-f', action="store", dest="config_file_path")
    args = parser.parse_args()
    config_file_path = args.config_file_path
    fp = open(config_file_path, "r")
    yaml_obj = yaml.safe_load(fp)
    hosts = yaml_obj['hosts']
    username = yaml_obj['username']
    # parent_dir = yaml_obj['trace_parent_dir']
    
    # fp = open("./utah.yaml", "r")
    # yaml_obj = yaml.safe_load(fp)
    # hosts_new = yaml_obj['hosts']
    # trace_path = parent_dir+"train-traces/"
    # output_path = parent_dir+"output-offline-3d/"
    # for line in open("/home/jane/Documents/MultiExpertHOCAdmission/train-list.out", "r"):
    #     trace_names.append(line.replace("\n", ""))
    # print(len(trace_names))
    # trace_path = parent_dir+"traces/"
    # output_path = parent_dir+"output/"
    
    # print('total_hosts:', len(hosts))
    # count_per_node = 700/len(hosts)
    # print(count_per_node)
    # transfer_features()
    # for host in range(0, 20):
    #     for rem in range(1, 6):
    #         total_list.append(str(host)+"-node"+str(rem))
    # print(total_list)

    # host_zip_num = math.floor(len(total_list)/len(hosts))
    # print(host_zip_num)
    # assigned_zip_num = 0
    # for i in range(0, len(hosts)):
        # replicate(i)
        # if hosts[i].startswith("cl") or hosts[i].startswith("c220"):
        #     host_zip_num = 2
        # else:
        #     host_zip_num = 1
        # replicate_to(i, assigned_zip_num, host_zip_num)
        # assigned_zip_num += host_zip_num
        # copy_data(i)
    for i in range(0, len(hosts)):
        host = hosts[i]
        print(host)
        # ssh_object = connect_rhost(host, username)
        # ssh_hosts.append(ssh_object)
        
        # if i >= 45:
        if host not in inactive_nodes:
            ssh_object = connect_rhost(host, username)
            ssh_hosts.append(ssh_object)
            # setup(i)
            # gen_correlation(i)
        # else:
        #     ssh_hosts.append(1)
        # # if i > 3:
        # #     unzip(i, assigned_zip_num, host_zip_num)
            check_zip(i)
        else:
            ssh_hosts.append(1)