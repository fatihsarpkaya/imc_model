#!/usr/bin/python3

#!/usr/bin/python

# First argument for sending behavior
# 2nd (optional): for additional buffersize (additional to bandwidth delay product)

from mininet.net import Mininet
from mininet.node import Host, CPULimitedHost
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController, DefaultController, OVSSwitch
from mininet.nodelib import LinuxBridge
from functools import partial
import subprocess
from mininet_experiments.experiment_topology import BottleneckTopology
from mininet_experiments.logparser import external_main as logparser_main
import re
import sys
import time
import math
import subprocess
import os
import threading
from datetime import datetime
import yaml
from collections import Counter
import argparse

logfolder = 'hostlogs/'
datafolder = 'hostdata/'
condenseddatafolder = 'condensed/'

# Measuring the queue length of the switch called 's1'.
# Start thread with `.start()`, stop with `.save_stop()` und dann `.join()`
# configuration: TC_QUEUE_SAMPLE_PERIOD (waiting time until next measurement)
class QueueMeasurements(threading.Thread):
    def __init__(self, output_file, device_name ,tc_queue_sample_period):
        threading.Thread.__init__(self)
        self.output_file = open(output_file, "w")
        self.stopped = False
        self.finished = False
        self.device_name = device_name
        self.tc_queue_sample_period = tc_queue_sample_period

    def safe_stop(self):
        self.stopped = True

    def run(self):
        print("Starting queue measurement thread.")
        queue_pattern = re.compile(r'backlog\s[^\s]+\s([\d]+)p')
        cmd = "tc -s qdisc show dev " + self.device_name
        while not self.stopped:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            output = p.stdout.read()
            #print(output)
            matches = queue_pattern.findall(str(output)) # Usually two matches: First match is HTB, Second is NetEm
            if matches == []:
                print("No match found.")
                break
            if len(matches) == 1:
                print("Only one match found. Untypical")
                self.output_file.write(("%.6f,%s" % time.time(), str(matches[0])))
            else:
                self.output_file.write("%.6f,%s,%s\n" % (time.time(), str(matches[0]), str(matches[1])))
            time.sleep(self.tc_queue_sample_period)

        print("Closing queue measurement thread.")
        self.output_file.close()
        self.finished = True

def disable_ipv6(net):
   # Disable IPv6. produces too much noise
   for h in net.hosts:
       h.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
       h.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
       h.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")

   for sw in net.switches:
       sw.cmd("sysctl -w net.ipv6.conf.all.disable_ipv6=1")
       sw.cmd("sysctl -w net.ipv6.conf.default.disable_ipv6=1")
       sw.cmd("sysctl -w net.ipv6.conf.lo.disable_ipv6=1")


def mininet_with_tcpdump(config):

    if config['controller'] == 'remote':
        startControllerCommand = [config['pox_directory']+'/pox.py', 'openflow.of_01', '--port=6634',  'forwarding.l2_learning']
        controllerProcess = subprocess.Popen(startControllerCommand, stdout=subprocess.PIPE,
           stderr=subprocess.STDOUT)

    time.sleep(2)

    linkLatency = config['link_latency']
    linkCapacity = config['link_capacity']
    sending_behavior = config['sending_behavior']

    print(config['behavior_command'])

    # Parametrize 
    if 'PCCFLEX' in config['behavior_command'] and 'utility_function_parameters' in config.keys():
        param_names = ['rate_sensitivity', 'rate_exponent', 'latency_sensitivity', \
                       'inflation_sensitivity', 'loss_sensitivity']
        for i in range(len(config['utility_function_parameters'])):
            with open('/sys/module/tcp_pcc/parameters/'+param_names[i], 'w') as param_file:
                if i == 0:
                    ufp_val = int(config['utility_function_parameters'][i] * 1000)
                else:
                    ufp_val = int(config['utility_function_parameters'][i])
                param_file.write("%d" % ufp_val)

    if config['source_latency']:
        random_lat = config['source_latency_range']
    else:
        random_lat = None

    topo = BottleneckTopology(srcHosts=len(sending_behavior), linkCapacity=linkCapacity, linkLatency=linkLatency,
                              bufferFactor=config['switch_buffer'],
                              bw_delay_product=config['inferred']['bw_delay_product'], random_latency_range=random_lat,
                              use_htb=config['use_htb'], use_red=config['use_red'])
    if config['controller'] == 'remote':
        controller = partial(RemoteController, ip='127.0.0.1', port=6634)
    elif config['controller'] == 'default':
        controller = DefaultController

    net = Mininet(
            topo=topo,
            controller=controller,
            autoSetMacs=True,
            switch=OVSSwitch,
            autoStaticArp=True,
            host=Host,
            link=TCLink
        )

    if config['disable_ipv6']:
        disable_ipv6(net)
    net.start()

    hDest = net.hosts[-1]
    destHostID = len(net.hosts)
    nSrcHosts = destHostID-1

    resultFilePrefix = config['result_dir']

    # Start destination host
    print("Starting host hDest")
    hDest.cmd('mininet_experiments/receiving_host.py '+str(destHostID)+' '+ config['result_dir'] + 'config.yaml > ' +
              resultFilePrefix +  logfolder + 'host'+ str(destHostID) + '.log 2>&1 &')

    dev_name = "s1-eth" + str(nSrcHosts + 1) # TODO: Rather give explicit name for NIC

    queue_meas_thread = QueueMeasurements(resultFilePrefix + "queue_length.csv", dev_name, config['tc_queue_sample_period'])
    queue_meas_thread.start()

    if 'BBR2' in config['behavior_command']:
        with open(resultFilePrefix + "bbr2_internals.log", 'w') as bbr2InternalsFile:
            dmesgCommand = ('dmesg --follow-new --time-format iso').split()
            subprocess.Popen(dmesgCommand, stdout=bbr2InternalsFile, stderr=bbr2InternalsFile)
        print('Started kernel logging for BBR2 debug output.')

    if 'PCCFLEX' in config['behavior_command']:
        with open(resultFilePrefix + "pcc_internals.log", 'w') as pccInternalsFile:
            dmesgCommand = ('dmesg --C').split()
            subprocess.Popen(dmesgCommand, stdout=pccInternalsFile, stderr=pccInternalsFile)
            dmesgCommand = ('dmesg --follow --time-format iso').split()
            subprocess.Popen(dmesgCommand, stdout=pccInternalsFile, stderr=pccInternalsFile)
        print('Started kernel logging for PCC debug output.')

    # Start sending on source hosts
    print("Invoking hosts...")
    for i in range(nSrcHosts):
        print("Host #%d started." % (i))
        net.hosts[i].cmdPrint('mininet_experiments/sending_host.py '+str(i)+' '+str(destHostID)+' ' + config['result_dir']  +
                              'config.yaml  > ' + resultFilePrefix + logfolder + 'host'+ str(i+1) + '.log 2>&1 &')

    num_samples = float(config['send_duration'])/float(config['memory_sampling_period'])
    for i in range(int(num_samples)):
        os.system("echo -n '" + str(time.time()) + ",' >> " + resultFilePrefix + "sysmemusage.csv")
        os.system("free -m | grep 'Mem' | awk -v OFS='\t' '{print $3}' >> " + resultFilePrefix + "sysmemusage.csv")
        os.system("\n >> " + resultFilePrefix + "sysmemusage.csv")
        time.sleep(config['memory_sampling_period'])
    else:
        time.sleep(int(config['send_duration']))
    print("Sending over: ", time.time())
    time.sleep(5)

    # Datacollection in Mininet

    # Stop measuring
    queue_meas_thread.safe_stop()
    queue_meas_thread.join()

    ## Safe-quit tcpdump
    hDest.cmd('pkill -SIGTERM -f tcpdump')
    hDest.cmd('pkill -9 -f "iperf -s"')

    #CLI(net)

    net.stop()
    if config['controller'] == 'remote':
        controllerProcess.kill()

# Go from behavior config dict to behavior summary string

# Help for parsing. Structure of sending behavior:
# 'sending_behavior': [{'h1': {'protocol': 'tcp-cubic', 'tso_on': False}},
#                       {'h2': {'protocol': 'udp-stable'}}]

def createBehaviorSummary(sending_behavior_dict, config):
    # Lists all protocols that are used
    protocols = [[a['protocol'] for a in client.values()][0] for client in sending_behavior_dict]

    #protocols =[ client['protocol'] for client in sending_behavior_dict]
    counts = Counter(protocols)
    summary = []
    for k in sorted(counts.keys()):
        summary.append(config['behavior_summary_mapping'][k] + '-' + str(counts[k]))
    ret = '_'.join(summary)
    return ret

# Parse the first input
# Create new sending behavior based on string and load it into config.
# Will support old parsing method. '_' for separating behavior types, '-' for parsing the number
# Example string: TCP-8_STABLE-2 will create 8 tcp and 2 udp clients
def parseBehaviorSummary(summaryString, config):
    types = summaryString.split('_')
    hosts = 1
    sending_behavior = []

    for combo in types:
        print(combo)
        if '-' in combo:
            type, num = combo.split('-')
        else:
            num = 1
            type = combo
        behavior = config['send_behavior_parsing'][type]
        for i in range(int(num)):
            sending_behavior.append({'h' + str(hosts): {'protocol': behavior}})
            hosts += 1
    config['sending_behavior'] = sending_behavior


## Result folders are generated growing in depth according to: SEND_DURATION, NSRCHOSTS,
#   LINK_CAPACITY, BUFFER_SIZE, BEHAVIOR_SUMMARY
def generateResultDir(behavior_summary, config):
    if config['base_res_dir']:
        resultDir = config['base_res_dir'] + 'results/'
    else:
        resultDir = 'results/'

    resultParam = [config['send_duration'], config['inferred']['num_senders'],  config['link_capacity'],
                   config['switch_buffer'], behavior_summary]
    for rP in resultParam:
        resultDir += str(rP)+'/'
        if not os.path.exists(resultDir):
            os.system('mkdir -p ' + resultDir)
    resultDir += datetime.strftime(datetime.now(), "%Y-%m-%d--%H-%M-%S") + '/'
    os.system('mkdir -p ' + resultDir)
    for rT in ['hostlogs/', 'hostdata/', 'condensed/']:
        os.mkdir(resultDir+rT)
    config['result_dir'] = resultDir
    return resultDir

def runExperiment(config):

    mininet_with_tcpdump(config)

    resultFilePrefix = config['result_dir']
    print("Initiating logparser: " + resultFilePrefix)
    logparser_main(resultFilePrefix)


# Use the default values for sending behavior to complete the sender configs
def completeSenderConfig(config):
    if not (config['behavior_command'] == None):
        parseBehaviorSummary(config['behavior_command'], config)
    send_defaults = config['defaults']['sending_behavior']
    for sender in config['sending_behavior']:
        if len(sender.keys()) > 1: # Should not happen.
            print("Unexpected number of keys for sender! Keys: ", sender.keys(), '\nConfig: ', config)
        for name in sender.keys():
            props = sender[name]
            # Protocol defaults
            if not 'protocol' in props:
                props['protocol'] = send_defaults['protocol']
            elif props['protocol'] == 'tcp':
                if not 'cc_flavour' in props:
                    props['protocol'] = 'tcp-' + send_defaults['cc_flavour']
                else:
                    props['protocol'] = 'tcp-' + props['cc_flavour']
            elif props['protocol'] == 'udp':
                if not 'udp_sending_behavior' in props:
                    props['protocol'] = 'udp-' + send_defaults['udp_sending_behavior']
                else:
                    props['protocol'] = 'udp-' + props['udp_sending_behavior']
            # TSO Default
            if 'tcp' in props['protocol'] and not 'tso_on' in props:
                props['tso_on'] = send_defaults['tso_on']
    return config



# Create configuration that is either hardcoded or inferred by the default_config
def inferConfig(config):
    inferred = {}
    inferred['num_senders'] = len(config['sending_behavior'])

    linkLatency = config['link_latency']
    linkCapacity = config['link_capacity']
    # packet_size = 1512 # Based on TCP dump of Iperf Traffic. But might need to consider ACK traffic too? (is buffer for both?)
    packet_size = 1514 #

    bw_delay_prod = int(math.ceil(((linkLatency / 1000) * (linkCapacity / 8) * 1e6) / packet_size))
    inferred['bw_delay_product'] = bw_delay_prod
    inferred['buffer_size'] = bw_delay_prod + int(bw_delay_prod * config['switch_buffer'])
    # Sizes as from wireshark and tcpdump
    packetsize_udp = 1512
    payload_udp = 1470

    if config['cbr_as_pss']:
        ppsrate = math.floor(config['link_capacity']*1000000 / (float(packetsize_udp)* 8))
        inferred['cbr_rate'] = ppsrate
    else:
        goodput_ratio_udp = float(payload_udp) / packetsize_udp
        inferred['cbr_rate'] =  goodput_ratio_udp * config['link_capacity'] / float(inferred['num_senders'])

    inferred['cbr_rate'] = inferred['cbr_rate'] * config['cbr_adjustment']

    inferred['behavior_summary'] = createBehaviorSummary(config['sending_behavior'], config)
    config['inferred'] = inferred
    return config

# If an explicit config is passed, will ignore any CLI arguments
def setup_configuration(args=None, explicit_config=None):

    if not explicit_config:
        if args.config_file:
            config_file = args.config_file
        else:
            config_file = "config-defaults.yaml"
    
        with open(config_file, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            # Load Arguments
            setLogLevel(config['mininet_log_level'])
    
    
        if args.summary:
            parseBehaviorSummary(args.summary, config)
        if args.switch_buffer:
            config['switch_buffer'] = int(args.switch_buffer)
        if args.tso_on:
            config['defaults']['sending_behavior']['tso_on'] = True
    else:
        config = explicit_config

    config = completeSenderConfig(config)

    # Infer Configuration
    config = inferConfig(config)
    # Create Result Directory
    resultFilePrefix = generateResultDir(config['inferred']['behavior_summary'], config)  # save it as: 'result_dir' config

    # Dump Config
    f = open(resultFilePrefix + 'config.yaml', 'w')
    yaml.dump(config, f)
    f.close()
    return config

def main(explicit_config=None):

    if not explicit_config:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, dest='config_file', help='Path to configuration file')
        parser.add_argument('--summary', type=str, dest='summary', help='Behavior summary string')
        parser.add_argument('--buffer', type=float, dest='switch_buffer', help='Switch-buffer size in BDP of link')
        parser.add_argument('--tso-on', dest='tso_on', help='Enable TSO.')
        args = parser.parse_args()
        config = setup_configuration(args=args)
    else:
        config = setup_configuration(explicit_config=explicit_config)

    runExperiment(config)
    print("Experiment finished.")
    print("Resultfolder: ", config['result_dir'])
    return config['result_dir']

if __name__ == "__main__":
    main()

