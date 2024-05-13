#!/usr/bin/python3

# Takes 3 arguments:
# hostID: provide an integer X, assuming the sending hostnames are 'hX'
# destHostID: provide the desthostID
# configloc: the location of the configfile (default file won't work. some by cc-experiment inferred values are needed)

import os
import sys
import subprocess
import re
import time
import random
import yaml
from pyroute2 import IPRoute

INTERVALS = 15
PROBING_INTERVAL = 1

NPATHS = 1

def initiateLog(hostID):
    if os.path.exists(config['result_dir']+"hostlogs/%s.log" % hostID):
        return
    with open(config['result_dir']+"hostlogs/%s.log" % hostID, "w+") as logfile:
        logfile.write(("%.6f" % time.time())+": "+hostID+": Started\n")

def log(hostID, logContent):
    with open(config['result_dir']+"hostlogs/%s.log" % hostID, "a+") as logfile:
        logfile.write(("%.6f" % time.time())+": "+hostID+": "+logContent+"\n")


def startTcpDump(hostID):
    for i in range(1):
        with open(config['result_dir']+'hostdata/'+str(hostID)+'-eth'+str(i)+'.log', 'w+') as f:
            tcpDumpCommmand = ('tcpdump -tt -i '+str(hostID)+'-eth'+str(i)+' -n -e -v -S -x -s 96').split()
            subprocess.Popen(tcpDumpCommmand, stdout=f, stderr=f)
            log(hostID, "Started tcpdump.")

def setupInterface(hostID, IPNum):
    ip = IPRoute()
    index = ip.link_lookup(ifname=''+hostID+'-eth1')[0]
    ip.addr('add', index, address='10.0.1.'+IPNum, mask=24)
    ip.close()
    log(hostID, "Second interface set up.")

def setTSO(hostID, on_mode):
    mode = "on" if on_mode else "off"
    for ifID in range(NPATHS):
        turnoffTSOCommand = ("ethtool -K %s-eth%d tso %s" % (hostID, ifID, mode)).split()
        output = str(subprocess.check_output(turnoffTSOCommand))
    log(hostID, "TSO turned " + str(mode))

def announceYourself(hostID, desthostID):
    for ifID in range(NPATHS):
        log(hostID, "Announce %s-eth%d" % (hostID,ifID))
        pingCommand = ("ping -c 3 -I %s-eth%d 10.0.%d.%d" % (hostID, ifID, ifID, desthostID)).split()
        subprocess.call(pingCommand)

# This will always be executed; regardless of 'protocol' config.
def iperf_command_base(currPath, desthostID, IPNum, duration, sampling_period, format):
    return ("iperf -c 10.0.%s.%d -B 10.0.%s.%s -t %d -i %s -e -f %s " % (currPath, desthostID, currPath, IPNum, duration, sampling_period, format)).split()

def tcp_command(cc_flavour, mss):
    return ("-p 5002 -Z %s -M %d " % (cc_flavour, mss)).split()


def tcp_command_paced(cc_flavour, mss, cbr_as_pps, cbr_rate):
    if cbr_as_pps:
        return ("-p 5002 -Z %s -M %d -b %spps" % (cc_flavour, mss, cbr_rate)).split()
    else:
        return ("-p 5002 -Z %s -M %d -b %sm" % (cc_flavour, mss, cbr_rate)).split()

#def useCSRCommand(currPath, hostID):
#    return ("iperf -c 10.0.%s.%d -p 5002 -B 10.0.%s.%s -t %d -w %sM " % (currPath, DESTHOSTID, currPath, hostID, IPERF_DURATION, CSR_RATE)).split()

def udp_stable_command(cbr_as_pps, cbr_rate):
    if cbr_as_pps:
        return ("-p 5003 -u -b %spps " % (cbr_rate)).split()
    else:
        return ("-p 5003 -u -b %sm " % (cbr_rate)).split()


def run(behavior_index, desthostID, config):
    print(config['sending_behavior'][behavior_index].keys())
    hostID, behavior = [(i, j) for i, j in config['sending_behavior'][behavior_index].items()][0]
    IPNum = behavior_index + 1
    initiateLog(hostID)
    log(hostID, ">> " + str(desthostID))
    announceYourself(hostID, desthostID)
    startTcpDump(hostID)
    random.seed(hostID)
    behavior = config['sending_behavior'][behavior_index][hostID] # TODO: make it compatible with custom hostnames
    log(hostID, "Sending behavior: " + str(behavior) )

    # If duel mode, might have to wait:
    # special delay for protocol duels. see config
    if config['inferred']['num_senders'] == 2 and behavior['protocol'] == config['goes_second'] and config['duel_delay'] != 0:
        time.sleep(config['duel_delay'])
    command = iperf_command_base(0, desthostID, IPNum, config['send_duration'], config['iperf_sampling_period'], config['iperf_outfile_format'])
    protocol = behavior['protocol']
    if 'tcp' in protocol:
        #reduceMTUCommand = ("ifconfig h%s-eth%d mtu 100" % (hostID, 0)).split()
        #subprocess.call(reduceMTUCommand)
        setTSO(hostID, behavior['tso_on'])
        if protocol == 'tcp-cubic':
            command += tcp_command('cubic', config['mss'])
        elif protocol == 'tcp-reno':
            command += tcp_command('reno', config['mss'])
        elif protocol == 'tcp-bbr':
            command += tcp_command('bbr', config['mss'])
        elif protocol == 'tcp-bbr2':
            command += tcp_command('bbr2', config['mss'])
    elif 'udp' in protocol:
        if protocol == "udp-stable":
            command += udp_stable_command(config['cbr_as_pss'], config['inferred']['cbr_rate'])
        else:
            print("Undefined UDP behavior.")
            return

    currPath = '0'

    iperfoutputfile = (config['result_dir'] + "hostlogs/" + config['iperf_outfile_client']).replace("$", str(IPNum))
    fout = open(iperfoutputfile, 'w')

    time.sleep(2)
    log(hostID, "Executing Command: " +  str(command))
    iperf_Process = subprocess.Popen(command, stdout=fout)
    cwind_period = float(config['cwind_sampling_period'])
    if 'tcp' in protocol:
        for i in range(int(config['send_duration'] / cwind_period)):
            time.sleep(cwind_period)
            ssOutput = str(subprocess.check_output('ss -ti'.split()))
            #log(hostID, ssOutput)
            m = re.match(r'.*(cwnd:\d+).*', ssOutput)
            if m is not None:
                log(hostID, m.group(1))
            if 'tcp-bbr' in protocol:
                m = re.match(r'.*bbr:\(bw:(\S+).bps.*mrtt:(\S+),pac.*(pacing_rate \S+).bps.*(delivery_rate \S+).bps.*', ssOutput)
                if m is not None:
                    log(hostID, 'btl_bw {} | mrtt {} | {} | {}'.format(m.group(1), m.group(2), m.group(3), m.group(4)) )
    else:
        time.sleep(config['send_duration'])
    iperf_Process.communicate()
    fout.close()
    log(hostID, "Host %s finished experiment" % hostID)

def parseargs():
    behavior_index = int(sys.argv[1])
    desthostID = int(sys.argv[2])
    configfile_location = sys.argv[3]
    return behavior_index, desthostID, configfile_location

if __name__ == "__main__":
    behavior_index, desthostID, configloc = parseargs()
    f = open(configloc, "r")
    config = yaml.safe_load(f)
    f.close()
    run(behavior_index, desthostID, config)

