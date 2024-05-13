#!/usr/bin/python3

from datetime import datetime, timezone
import os
import sys
import re
import subprocess
import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
from mininet_experiments.plotting import *
# from mininet_experiments.plotting_single import *
#from plotting import *
#from plotting_single import *


import pprint

# Notes:
# Now the statistics are gathered when processing the raw data.
# So far, there is only cropping available after the raw data processing, which messes up the stats.
# TODO: Allow for cropping that is applied during raw data traversal.
# --> this will probably make the loadCondensedData obsolete, which I think is not problem,
#       since the processing times were never that long anyway that it would require an intermediate repr.

# Note:
# The statistics gathering is not separated into origin (which could be relevant for the host)
# there could be a benefit in restructuring that at some later point.

logfolder = 'hostlogs/'
datafolder = 'hostdata/'
condenseddatafolder = 'condensed/'

TIMEAGGREGATION = 1  # Resolution of timestamps,  '1' rounds to 10ths of seconds, '2' rounds to 100ths, etc.
SMOOTHING       = 1

RESULT_FILE_PREFIX = ''


MERGE_INTERVALS = [[1,10]]#, [2,10]]

ALL_FLOWS = ['10.0.0.%d' % i for i in range(1,10+1)]
PLOT_KEYS = ['10.0.0.1', 'x.1-10'] #, 'x.2-10']
PLOT_CWND_KEYS = ['10.0.0.1']

SUM        = 'SUM'
MAX        = 'MAX'
AVG        = 'AVG'
VAR        = 'VAR'


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{amssymb}'


def inflight_calc(data, econfig):
    data.loc[:, 'inflight_sum'] = 0
    data.loc[:, 'bytes_acked'] = 0
    num_senders = econfig['inferred']['num_senders']
    for s in range(num_senders):
        sender = s+1
        sender_data = data[data.measuredon == str(sender)]
        last_byte_sent = -1
        last_byte_acked = -1
        for r, row in sender_data.iterrows():
            if row['src'] != num_senders+1:
                last_byte_sent = row['seqno']
                data.at[r, 'inflight_sum'] = max(last_byte_sent + row['payload'] - last_byte_acked, 0) if last_byte_acked != -1 else 0
            else:
                data.at[r, 'bytes_acked'] = max(row['ackno'] - last_byte_acked, 0)
                last_byte_acked = row['ackno']

    print("Calculated inflight.")


# INPUT:
# key is either 'udpno' or 'seqno'
# Note: seqno != -1 => udpno == -1 && udpno != -1 => seqno == -1
# Requirement: the data passed contains only entries with 'identificator'-value != -1
#   Therefore: can not be a mix between UDP and TCP
# Requirement: the data passed is outbound traffic from sender to destination.
#   Therefore: ACK loss and latency is not calculated.
# OUTPUT:
# Losses are registered at timestamp where lost packet was sent.
# Latency is registered at the sender timestamp.

def loss_calc(received, sent, key):
    more_output = False

    if received.shape[0] == 0 or sent.shape[0] == 0:
        return received, sent
    data = sent.append(received)
    data.sort_values(by=[key, 'timestamp'], inplace=True)
    #print("Concatenated and sorted: ")
    #print(data)

    # Only for Seqno: Find duplicate keys in sender, mark not-last ones as losses
    if key == 'seqno':
        data.loc[(data.measuredon != 'Dest') & ((data[(data.measuredon != 'Dest')]).duplicated(subset=key, keep='last')), 'loss'] = 1

        # Technically, there should not be duplicates on the receiver side, but for some reason this happens. Is it when the resending is due to timeout?
        # We will keep track of it as well.
        data.loc[(data.measuredon == 'Dest') & (data[(data.measuredon == 'Dest')].duplicated(subset=key, keep='last')), 'double_receive'] = 1
        data.loc[(data.measuredon == 'Dest') & (data[(data.measuredon == 'Dest')].duplicated(subset=key, keep='last')), 'loss'] = 1 # Only for avoidng

    # Safety Check: No duplicate keys left among sender or receiver
    #if ~data.duplicated(subset='udpno', keep=False):
    dest_has_dupl = data[(data.loss == 0) & (data.measuredon == 'Dest')].duplicated(subset=key, keep=False).any()
    sender_has_dupl = data[(data.loss == 0) & (data.measuredon != 'Dest')].duplicated(subset=key, keep=False).any()
    if dest_has_dupl or sender_has_dupl:
        print("Problem: Duplicates of ", key, " in dest/sender: " +  str(dest_has_dupl) + "/" + str(sender_has_dupl))
        num_dest_dupl = np.count_nonzero(data[(data.loss == 0) & (data.measuredon == 'Dest')].duplicated(subset=key, keep=False))
        num_src_dupl = np.count_nonzero(data[(data.loss == 0) & (data.measuredon != 'Dest')].duplicated(subset=key, keep=False))
        if num_dest_dupl == num_src_dupl:
            print("But numbers are identical, so it's prob. ok.")
        else:
            print("Nonzeros in dest: ", num_dest_dupl)
            print("Nonzeros in sender: ", num_src_dupl)
            print(data[(data.loss == 0) & (data.measuredon == 'Dest')].duplicated(subset=key, keep=False))
            print(data[(data.loss == 0) & (data.measuredon != 'Dest')].duplicated(subset=key, keep=False))
            raise Exception

    # Now all remaining duplicates that are loss == 0 are the sent-received pair, therefore acked.
    # Therefore: find nonacked pairs/nonduplicates and mark them as losses.
    # TODO: nuance, safety check: check for nonduplicates on receiver side.
    data.loc[((data.loss == 0) & ~(data[(data.loss == 0)].duplicated(subset=key, keep=False))), 'loss'] = 1

    # Safety check: All non-lost packets are acked, therefore shapex should be the same
    receiver_shape = data[(data.loss == 0) & (data.measuredon == 'Dest')].shape
    sender_shape = data[(data.loss == 0) & (data.measuredon != 'Dest')].shape

    if more_output:
        print("Shapes: ", receiver_shape, " and ", sender_shape)
    if receiver_shape[0] != sender_shape[0]:
        print("Non-lost samples on receiver and sender do not match! Sender: ", sender_shape[0], " receiver: ", receiver_shape[0])
        with pd.option_context('display.max_rows', None):  # more options can be specified also
            #print(data)
            raise Exception
    eq = data[(data.loss == 0) & (data.measuredon == 'Dest')][key].values == data[(data.loss == 0) & (data.measuredon != 'Dest')][key].values
    if not eq.all():
        print("Equal length but not equal! ", eq)
        raise Exception

    # Logic
    data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'num'] = 1
    data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'latency_sum'] = \
        data.loc[(data.loss == 0) & (data.measuredon == 'Dest'), 'timestamp'].values - \
        data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'timestamp'].values
    data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'jitter_sum'] = \
        np.abs(data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'latency_sum'].values - \
        data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'latency_sum'].shift(1).values)
    data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'jitter_sum_sq'] = \
            np.square(data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'jitter_sum'].values)

    #print(data.loc[(data.loss == 0) & (data.measuredon != 'Dest'), 'jitter_sum'])

    if key == 'seqno':
        data.loc[(data.measuredon == 'Dest') & (data[(data.measuredon == 'Dest')].duplicated(subset=key, keep='last')), 'loss'] = 0 # Only for avoidng

    return data[(data.measuredon == 'Dest')], data[(data.measuredon != 'Dest')]

# Columns: timestamp, measuredon, src, dest, load, payload, udpno, seqno, ackno
def processTCPDdata(filename, econfig, timestep=0.1):
    df = pd.read_csv(filename, dtype={'timestamp': np.float64, 'measuredon': 'str', 'src': np.int64,
                                      'dest': np.int64, 'load': np.int64, 'payload': np.int64,
                                      'udpno': np.int64, 'seqno': np.int64, 'ackno': np.int64, 'id': np.int64,
                                      'udpno': np.int64, 'seqno': np.int64, 'ackno': np.int64, 'id': np.int64,
                                      })
    num_senders = econfig['inferred']['num_senders']
    receiver_no = num_senders + 1 # Todo: it's weird that sometimes 'Dest' is used and sometimes IP '11'. standardize!

    inflight_calc(df, econfig)

    # For some reason, Pandas does not accept assignment among non-strict subsets, 
    # e.g., df[(df.updno != -1)] =  df[(df.updno != -1)] if df == df[(df.updno != -1)]
    # Hence, we add this filler_df, which is a weird fix, but it works
    filler_df = pd.DataFrame({'timestamp': [-1], 'measuredon': ['-1'], 'src': [-1],
                              'dest': [11], 'load': [0], 'payload': [0],
                              'udpno': [-1], 'seqno': [-1], 'ackno': [-1], 'id': [-1]})

    resampled = []
    for s in range(num_senders):

        sender = s + 1
        received_from_sender = df[(df.src == sender) & (df.dest == receiver_no) & (df.measuredon == 'Dest')]
        received_from_sender.loc[:, 'loss'] = 0  # Does not contribute to lossstat, but is used in loss_calc

        sent_by_sender = df[(df.measuredon == str(sender)) & (df.src == sender) & (df.dest == receiver_no)].copy()
        filler_df.loc[0, 'timestamp'] = sent_by_sender['timestamp'].iloc[0]
        sent_by_sender = sent_by_sender.append(filler_df)
        sent_by_sender.loc[:, 'loss'] = 0
        sent_by_sender.loc[:, 'num'] = 0
        sent_by_sender.loc[:, 'latency_sum'] = 0.0  # In the beginning, it is the packet latency. Only later it is summed up.
        sent_by_sender.loc[:, 'jitter_sum'] = 0.0 # In the beginning, it is the packet jitter. Only later it is summed up.
        sent_by_sender.loc[:, 'jitter_sum_sq'] = 0.0
        received_from_sender.loc[:, 'double_receive'] = 0
        if econfig['more_output']:
            print("Shape:", received_from_sender.shape, " ", sent_by_sender.shape)
        try:
            received_from_sender[(received_from_sender.seqno != -1)], sent_by_sender[(sent_by_sender.seqno != -1)] = \
                loss_calc(received_from_sender[(received_from_sender.seqno != -1)], sent_by_sender[(sent_by_sender.seqno != -1)], 'seqno')
            _, sent_by_sender[(sent_by_sender.udpno != -1)] = \
                loss_calc(received_from_sender[(received_from_sender.udpno != -1)], sent_by_sender[(sent_by_sender.udpno != -1)], 'udpno')

        except:
            print("Error calculating loss and latency.")
            raise Exception

        # Senderside contributes: loss, latency_sum, latency_contributor_count
        sent_by_sender = sent_by_sender.filter(items=['timestamp', 'load', 'loss', 'latency_sum', 'num', 'jitter_sum', 'jitter_sum_sq', 'inflight_sum'])
        sent_by_sender = sent_by_sender.rename(columns={'load': 'load_sent'})
        sent_by_sender.loc[:, 'timestamp'] = pd.to_datetime(sent_by_sender.loc[:, "timestamp"], unit='s') # Need datetimeformat for resampling
        sent_by_sender = sent_by_sender.set_index('timestamp').resample(str(1000 * timestep) + 'ms', label='right').sum()
        sent_by_sender = sent_by_sender.add_suffix('_' + str(sender))
        resampled.append(sent_by_sender)
        if econfig['more_output']:
            print("Loss + Lat  + Jitter worked fine.")

        # Receiverside contributes: load, payload, number of double received packets.
        received_from_sender = received_from_sender.filter(items=['timestamp', 'load', 'payload', 'double_receive'])
        received_from_sender['timestamp'] = pd.to_datetime(received_from_sender["timestamp"], unit='s') # Need datetimeformat for resampling
        received_from_sender = received_from_sender.set_index('timestamp').resample(str(1000*timestep) + 'ms', label='right').sum()
        received_from_sender = received_from_sender.add_suffix('_' + str(sender))  # To make it distinguishable in table
        resampled.append(received_from_sender)

        # Backflow
        received_by_sender = df[(df.src == receiver_no) & (df.dest == sender) & (df.measuredon == str(sender))]
        received_by_sender = received_by_sender.filter(items=['timestamp', 'bytes_acked'])
        received_by_sender['timestamp'] = pd.to_datetime(received_by_sender["timestamp"], unit='s') # Need datetimeformat for resampling
        received_by_sender = received_by_sender.set_index('timestamp').resample(str(1000*timestep) + 'ms', label='right').sum()
        received_by_sender = received_by_sender.add_suffix('_' + str(sender))  # To make it distinguishable in table
        resampled.append(received_by_sender)


    load_table = pd.concat(resampled, axis=1, join='outer')
    load_table = load_table.reset_index()

    # Store it in epochs (unix seconds)
    load_table['abs_ts'] = load_table['timestamp'].values.astype(np.int64) / 1e9
    if econfig['more_output']:
        print("Absts appearance: ")
        print(load_table['abs_ts'])
    load_table['timestamp'] = (load_table.timestamp - load_table.timestamp.loc[0])  # Convert absolute time to timediff
    load_table['timestamp'] = load_table.timestamp.dt.total_seconds() # Elapsed seconds since start of experiment
    #print(load_table)
    load_table = load_table.sort_values(by=['timestamp'])
    load_table = load_table.set_index('timestamp')
    return load_table

#--------------------------------------------------------------------------------
# Parse and merge all tcpdump files
# Store in csv file.
# Fields: timestamp, measured-on, from, to, load, payload, udp, seqno, ackno

def parseTCPDumpMininet(datafiles, filedestination):
    # timestamp, measuredon, src, dest, load, payload, udpno, seqno, ackno
    more_output = False
    data = []
    data.append(['timestamp', 'measuredon', 'src', 'dest', 'load', 'payload', 'udpno', 'seqno', 'ackno', 'id'])
    for dfname in datafiles:

        measured_on = re.match(r'h(.+)-.*', dfname).group(1)
        datafile = RESULT_FILE_PREFIX+datafolder+dfname
        if more_output:
            print("Parsing datafile "+datafile+"...")

        wcOutput = str(subprocess.check_output(("wc -l "+datafile).split()))
        filelength = int(re.match(r'b\'(\d+).+', wcOutput).group(1))
        linecounter = 0

        with open(datafile, 'r') as df:
            linestring = '_'
            while(linestring):
                linestring = df.readline()
                linecounter += 1

                # Show progress
                if more_output:
                    if linecounter % 100000 == 0:
                        print("Read %d / %d lines." % (linecounter, filelength), end="\r")

                timestampMatcher = re.match(r'(\d+\.\d+).+\>.+', linestring)
                packetsizeMatcher = re.match(r'.+,\slength\s(\S+):.+', linestring)

                if (timestampMatcher and packetsizeMatcher): # If packet with timestamp and length:
                    try:
                        timestamp = timestampMatcher[1]
                        load = packetsizeMatcher[1]
                        id = int(re.match(r'.+,\sid\s(\d+).+', linestring).group(1))
                        offset = re.match(r'.+offset\s(\d+),.+', linestring)
                        if offset is None or int(offset.group(1)) != 0:
                            print("WARNING: Offset is not 0! ", linestring) # If this happens, it's a sign of fragmentation,
                                                                            # We should rethink the use of ID.
                        linestring = df.readline() # Proceed to second line of packet
                        linecounter += 1

                        hostOriginMatch = re.match(r'.*10\.0\.\d\.(\d+)\.\S+\s\>', linestring)
                        hostDestinationMatch = re.match(r'.+\>\s10\.0\.\d\.(\d+)\.\S+', linestring)
                        source = hostOriginMatch[1]
                        destination = hostDestinationMatch[1]
                        payload = int(re.match(r'.+,\slength\s(\S+)', linestring).group(1))

                        # Timeaggregation defines the granularity of the timestamps.
                        # timestamp = float(('%.'+str(TIMEAGGREGATION)+'f') % float(timestamp)) # Timestamp resolution
                        udpMatch = re.match(r'.+UDP.+', linestring)
                        sequenceMatch = re.match(r'.+seq\s(\d+):\d+.+', linestring) # Only capture right part of seqno range
                        # Assumption: only need right seqno. correct since iperf has consistent packet sizes.
                        #seqenceMatch = re.match(r'.+seq\s(\d+):(\d+).+', linestring)
                        ackedNrMatch = re.match(r'.+ack\s(\d+),.+', linestring)
                        if sequenceMatch:
                            seqno = int(sequenceMatch[1])
                        else:
                            seqno = -1
                        if ackedNrMatch:
                            ackno = ackedNrMatch[1]
                        else:
                            ackno = 0

                        # Parsing hexdump, depending if UDP or not
                        if udpMatch:
                            linestring = df.readline()
                            linestring = df.readline()  # Proceed to fourth line of packet
                            linecounter += 2
                            udpcMatch = re.match(r'\s*0x0010:\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)', linestring)
                            udpno = int(udpcMatch[1] + udpcMatch[2], 16)
                        else:
                            udpno = -1
                        if sequenceMatch and udpMatch:
                            print("Sequence AND UDP. Weird!")

                        line = [timestamp, measured_on, source, destination, load, payload, udpno, seqno, ackno, id]
                        data.append(line)
                    except:
                        if re.match(r'.+ICMP.+', linestring) is not None:  # ICMP is ok.
                            continue
                        else: # Else: Print
                            print("FAIL when parsing: ", linestring)
            print("Read all %d lines.                     " % (filelength))

        # Write compressed data to a csv file
        np.savetxt(filedestination, np.array(data), delimiter=",", fmt='%s')




#--------------------------------------------------------------------------------
# Parse raw load data files
def parseCwndFiles(datafiles):
    more_output = False
    print("Parsing CWND. Files: ", len(datafiles))

    cwndData = {}
    ssthreshData = {}

    for df in datafiles:

        #print("File: ", df)
        m = re.match(r'h(\d+).*', df)
        if not m:
            continue
        hostNr = m.group(1)
        ip = '10.0.0.'+hostNr

        datafile = RESULT_FILE_PREFIX+logfolder+df
        if more_output:
            print("Parsing datafile "+datafile+"...")

        cwndData[ip]     = {}
        ssthreshData[ip] = {}

        with open(datafile, 'r') as df:

            linestring = '_'
            while(linestring):
                linestring = df.readline()

                dataField = re.match(r'(\S+):.*cwnd:(\d+)', linestring)

                if dataField:
                    timestamp = float(dataField.group(1))
                    length = int(dataField.group(2))
                    cwndData[ip][timestamp] = length
                    continue

                dataField = re.match(r'(\S+):.*unacked:(\d+)', linestring)

                if dataField:
                    timestamp = float(dataField.group(1))
                    length = int(dataField.group(2))
                    ssthreshData[ip][timestamp] = length

    return cwndData, ssthreshData


# Assuming file structure (csv): unix-timestamp,packets-in-queue
def readQueueFile(datafile):
    print("Parsing queuefile.")

    headers = ['ts', 'queue1', 'queue']
    dtypes = {'ts': np.float64, 'queue': np.int64}
    df = pd.read_csv(datafile, header=None, names=headers, dtype=dtypes)

    ts_column = df['ts']
    queue_column = df['queue']
    return ts_column, queue_column



def calculateTotal(df, num_senders):
    df['total_load'] = (df[['load_' + str(i+1) for i in range(num_senders)]]).sum(axis=1)
    df['total_payload'] = (df[['payload_' + str(i+1) for i in range(num_senders)]]).sum(axis=1)
    return df
    #TODO df['total_latency']


#--------------------------------------------------------------------------------
# Get load data
def calculateLoad(econfig):

    parsed_data = RESULT_FILE_PREFIX + condenseddatafolder + 'tcpdump.csv'
    if not os.path.exists(parsed_data):
        datafiles = [f for f in os.listdir(RESULT_FILE_PREFIX + datafolder)]
        parseTCPDumpMininet(datafiles, parsed_data)

    condensed_data_file = RESULT_FILE_PREFIX + condenseddatafolder + 'tcpd_dataframe.csv'

    if not os.path.exists(condensed_data_file):
        datatable = processTCPDdata(parsed_data, econfig, econfig['plot_load_resolution'])
        datatable = calculateTotal(datatable, econfig['inferred']['num_senders'])
        datatable.to_csv(condensed_data_file)
    else:
        print("Loading condensed: ", condensed_data_file)
        datatable = pd.read_csv(condensed_data_file, dtype=np.float64)
        datatable = datatable.set_index("timestamp")


    # Truncate
    exp_duration = econfig['send_duration']
    truncate_front = econfig['truncate_front']
    truncate_back = econfig['truncate_back']
    datatable = datatable.truncate(before=truncate_front, after=exp_duration - truncate_back)

    return datatable


def loadExperimentConfig(resultpath):
    print(resultpath)
    f = open(resultpath + 'config.yaml')
    config = yaml.safe_load(f)
    f.close()
    return config



def loadFromCSV(filename):
    f = open(filename, 'r')
    df = pd.read_csv(f, header=None)
    return df


def main(savePlot=False):

    econfig = loadExperimentConfig(RESULT_FILE_PREFIX)
    econfig['more_output'] = False
    if not econfig['more_output']:
        import warnings
        warnings.simplefilter("ignore")

    print("============\nStarting with: ", RESULT_FILE_PREFIX, "\n==============")
    tcpd_data = calculateLoad(econfig)
    cwndData, ssthreshData = parseCwndFiles([f for f in os.listdir(RESULT_FILE_PREFIX+logfolder)])
    memdata = loadFromCSV(RESULT_FILE_PREFIX + "sysmemusage.csv")

    startTimestamp = tcpd_data.index.values[0]
    endTimestamp = tcpd_data.index.values[-1]

    startAbsTs = tcpd_data['abs_ts'].values[0]
    endAbsTs = tcpd_data['abs_ts'].values[-1]
    queueTs, queueVal = readQueueFile(RESULT_FILE_PREFIX + "queue_length.csv")
    print("Start/End timestamp: ", startTimestamp, endTimestamp)
    num_axes = sum([econfig['plot_loss'], econfig['plot_throughput'], econfig['plot_jitter'],
                    econfig['plot_cwnd'], econfig['plot_latency'], econfig['plot_queue'], econfig['plot_memory'],
                   2 * econfig['plot_iperf_losslat']])

    plt.figure('overview', figsize=(2, 2 * num_axes))

    fig, axes = plt.subplots(nrows=num_axes, num='overview', ncols=1, sharex=True, figsize=(100,4))

    xticks = []
    stats = {}
    ax_indx = 0
    figurename = "overview"
    if econfig['plot_throughput']:
        stats.update(plotLoad(figurename, axes[ax_indx], tcpd_data, startTimestamp, endTimestamp, econfig))
        ax_indx += 1
    if econfig['plot_cwnd']:
        stats.update(plotCwnd(figurename, axes[ax_indx], cwndData, ssthreshData, startAbsTs, endAbsTs, xticks))
        ax_indx += 1

    if econfig['plot_queue']:
        bdp = econfig['inferred']['bw_delay_product']
        real_buffer_size = bdp + int(econfig['switch_buffer'] * bdp)
        stats.update(plotQueue(figurename, axes[ax_indx], queueTs, queueVal, startAbsTs, endAbsTs, econfig['inferred']['bw_delay_product'],
                               real_buffer_size , xticks, econfig))
        ax_indx += 1
    if econfig['plot_latency']:
        stats.update(plotLatency(figurename, axes[ax_indx], tcpd_data, startTimestamp, endTimestamp, econfig))
        ax_indx += 1
    if econfig['plot_jitter']:
        stats.update(plotJitter(figurename, axes[ax_indx], tcpd_data, startTimestamp, endTimestamp, econfig))
        ax_indx += 1
    if econfig['plot_loss']:
        stats.update(plotLoss(figurename, axes[ax_indx], tcpd_data, startTimestamp, endTimestamp, econfig))
        ax_indx += 1
    if econfig['plot_memory']:
        stats.update(plotMemory(figurename, axes[ax_indx], memdata, startAbsTs, endAbsTs, econfig))
        ax_indx += 1

    print("Opening permissions...")
    os.system("sudo chmod -R 777 " +  RESULT_FILE_PREFIX)
    print("Parsing and plotting finished.")
    plt.close('all')


def external_main(resultfile_prefix):
    plt.figure('overview')
    plt.clf()
    global RESULT_FILE_PREFIX
    RESULT_FILE_PREFIX = resultfile_prefix
    main(savePlot=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plt.figure('overview')
        plt.clf()
        RESULT_FILE_PREFIX = sys.argv[1]
        main(savePlot=True)
    else:
        print("Please provide a result folder.")
