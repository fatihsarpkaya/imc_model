#!/usr/bin/python3

import sys
import time
import subprocess
from pyroute2 import IPRoute
import yaml
import os

RESULT_FILE_PREFIX = ''

def initiateLog():
    if os.path.exists(RESULT_FILE_PREFIX+"hostlogs/hDest.log"):
        return
    with open(RESULT_FILE_PREFIX+"hostlogs/hDest.log", "w+") as logfile:
        logfile.write(("%.6f" % time.time())+": hDest: Started\n")


def log(logContent):
    with open(RESULT_FILE_PREFIX+"hostlogs/hDest.log", "a+") as logfile:
        logfile.write(("%.6f" % time.time())+": hDest: "+logContent+"\n")


def startTcpDump():
    # > '+RESULT_FILE_PREFIX+'hostdata/hDest-eth'+str(i)+'.log
    for i in range(1):
        with open(RESULT_FILE_PREFIX+'hostdata/hDest-eth'+str(i)+'.log', 'w+') as f:
            tcpDumpCommmand = ('tcpdump -tt -i hDest-eth'+str(i)+' -e -v -n -S -x -s 96').split()
            subprocess.Popen(tcpDumpCommmand, stdout=f, stderr=f)
            log("Started tcpdump.")


# Start iperf server with TCP on destination hosts
def startTcpServer(config):
    resfile = config['result_dir'] + config['iperf_outfile_server_tcp']
    samplingperiod = config['iperf_sampling_period_server']
    fout = open(resfile, "w")
    tcpIperfCommand = ('iperf -s -p 5002 -e -i %d -t %d -f %s' % (samplingperiod, config['send_duration'] + 5, config['iperf_outfile_format'])).split()
    print(tcpIperfCommand)
    log("Starting TCP Server.")
    log("Command: " + str(tcpIperfCommand))
    proc = subprocess.Popen(tcpIperfCommand, stdout=fout)
    return proc, fout

# Start iperf server with UDP on destination hosts
def startUdpServer(config):
    resfile = config['result_dir'] + config['iperf_outfile_server_udp']
    samplingperiod = config['iperf_sampling_period_server']
    fout = open(resfile, "w")
    udpIperfCommand = ('iperf -s -p 5003 -u -e -i %d -t %d -f %s' % (samplingperiod, config['send_duration'] + 10, config['iperf_outfile_format'])).split()
    print(udpIperfCommand)
    log("Starting UDP Server.")
    log("Command: " + str(udpIperfCommand))

    proc = subprocess.Popen(udpIperfCommand, stdout=fout)
    return proc, fout


# Setup second interface (only needed in multipath cases)
def setupInterface(hostID):
    ip = IPRoute()
    index = ip.link_lookup(ifname='hDest-eth1')[0]
    ip.addr('add', index, address='10.0.1.'+hostID, mask=24)
    ip.close()

def main(config, desthostID):
    global RESULT_FILE_PREFIX
    RESULT_FILE_PREFIX = config['result_dir']
    initiateLog()

    # Announce yourself
    pingCommand = ("ping -c 3 -I %s-eth%d 10.0.%d.%d" % ("hDest", 0, 0, 1)).split()
    subprocess.call(pingCommand)

    use_tcp = False
    use_udp = False
    protocols = [[a['protocol'] for a in client.values()][0] for client in config['sending_behavior']]
    for prot in protocols:
        if "tcp" in prot:
            use_tcp = True
        if "udp" in prot:
            use_udp = True

    log("Use TCP: " + str(use_tcp) + " | Use UDP: " + str(use_udp))
    startTcpDump()
    if use_tcp:
        tcp_proc, tcp_f = startTcpServer(config)
    if use_udp:
        udp_proc, udp_f = startUdpServer(config)
    if use_tcp:
        tcp_proc.communicate()
        tcp_f.close()
        log("Finished TCP Server.")

    if use_udp:
        udp_proc.communicate()
        udp_f.close()
        log("Finished UDP Server.")


def parseargs():
    desthostID = int(sys.argv[1])
    configfile_location = sys.argv[2]
    return desthostID, configfile_location

if __name__ == "__main__":
    desthostID, configloc = parseargs()
    f = open(configloc, "r")
    config = yaml.safe_load(f)
    f.close()
    main(config, desthostID)