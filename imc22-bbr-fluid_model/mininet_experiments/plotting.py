#!/usr/bin/python3.6

import re
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from colour import Color

logfolder = 'hostlogs/'
datafolder = 'hostdata/'
condenseddatafolder = 'condensed/'

RESULT_FILE_PREFIX = ''

COLORS = {
    'h1': {
        0: (0.0, 0.5, 1.0),
        1: (0.54, 0.81, 0.94)
    },
    'h2': {
        0: (0.8, 0.0, 0.0),
        1: (0.91, 0.45, 0.32)
    },
    'h3': {
        0: (0.0, 0.5, 0.0),
        1: (0.55, 0.71, 0.0)
    },
    'h4': {
        0: (0.5, 0.0, 0.1),
        1: (0.55, 0.71, 0.0)
    },
    'h5': {
        0: (0.2, 0.1, 0.6),
        1: (0.55, 0.71, 0.0)
    },
    'h6': {
        0: (0.5, 0.5, 1.0),
        1: (0.54, 0.81, 0.94)
    },
    'h7': {
        0: (0.5, 0.0, 0.0),
        1: (0.91, 0.45, 0.32)
    },
    'h8': {
        0: (0.5, 0.5, 0.0),
        1: (0.55, 0.71, 0.0)
    },
    'h9': {
        0: (1.0, 0.0, 0.1),
        1: (0.55, 0.71, 0.0)
    },
    'h10': {
        0: (0.7, 0.1, 0.6),
        1: (0.55, 0.71, 0.0)
    },
    'h11': {
        0: (0.7, 0.1, 0.6),
        1: (0.55, 0.71, 0.0)
    },
    'h2-10': {
        0: (0.7, 0.1, 0.6),
        1: (0.55, 0.71, 0.0)
    },
    'h1-10': {
        0: (1.0, 0.0, 0.1),
        1: (0.55, 0.71, 0.0)
    },
}

PLOT_CWND_KEYS = ['10.0.0.1']

def plotLoad(figurename, ax, tcpd_data, starttime, endtime, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()
    num_senders = econfig['inferred']['num_senders']
    linkCap = float(econfig['link_capacity'])
    emulated_bufferCap = int(econfig['inferred']['bw_delay_product'] * float(econfig['switch_buffer']))
    buffer_factor = float(econfig['switch_buffer'])
    real_bufferCap = econfig['inferred']['bw_delay_product'] + emulated_bufferCap
    sendBehav = econfig['inferred']['behavior_summary']
    sendBehav = sendBehav.replace('_', ' ')

    effective_duration = endtime - starttime
    converter = float(8.0 / 1e6)  # From bytes to Mbits
    timestep = econfig['plot_load_resolution']

    #print(tcpd_data.total_load[starttime:endtime])

    total_throughput = converter * np.sum(tcpd_data.total_load[starttime:endtime]) / float(effective_duration) # TODO redundant field: total_load
    total_load = 0
    capacity = float(linkCap)
    fair_share = capacity/ float(num_senders)

    ax.hlines([fair_share], starttime, endtime, linestyles='-.', label='Fair Share', colors='orange')
    ax.hlines([capacity], starttime, endtime, linestyles='-.', label='Link Bandwidth Capacity', colors='green')

    sender_throughputs = []
   # print(tcpd_data.columns)
    for hostid in list(range(1, num_senders+1)):
        label = "load_" + str(hostid)
        if label in tcpd_data.columns:
            total_load += np.sum(tcpd_data[label][starttime:endtime])
            average = np.sum(tcpd_data[label][starttime:endtime]) * converter / effective_duration
            sender_throughputs.append(average)
            if hostid in econfig['plot_hosts']:
                ax.plot(tcpd_data.index.values, converter * tcpd_data[label].values / timestep , ":", label=("h%s Throughput" % (hostid)))

    ax.plot(tcpd_data.index.values, converter * tcpd_data['total_load'].values / timestep , ':', label="Total Throughput")
    ax.set_title(r'\textbf{Flows:} ' + str(sendBehav) + r', \textbf{Capacity} ' + str(linkCap) + r'Mbps' + r', \textbf{Buffering}: ' + str(buffer_factor) + " * BDP")
    ax.hlines([total_throughput], starttime, endtime, linestyles='-.', label='Average Throughput', colors='blue')
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Rate (Mbps)')
    plt.locator_params(axis='both', nbins=20)
    # ax.set_xticks(xticks)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc=1)

    # Also make distribution plot
    avg_flow = np.average(sender_throughputs)
    plotDistribution(sender_throughputs, econfig['result_dir'] + 'throughput_dist.png',
                     lines=[avg_flow, fair_share],
                     labels=['Average Throughput', 'Fair Share'], colors=['blue', 'green'])

    # Return stats that were calculated here
    jain_fairness_index = jain_fairness(sender_throughputs, fair_share)
    if destination is not None:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(destination, dpi=300)
        plt.close(figurename)

    return {'total_throughput': float(total_throughput), 'utilization': total_throughput / linkCap,
            'timedelta': float(endtime - starttime), 'jain_fairness_index': float(jain_fairness_index),
            'throughput_average': float(avg_flow)}


# Calculate fairness index by jain
def jain_fairness(throughputs, fairshare):
    n = len(throughputs)
    optimals = n * [fairshare]  # Change this line for more general case of not everyone having same share.
    relative_allocations = np.divide(throughputs, optimals) # Divide all throughputs by their optimal throughput
    numerator = np.square(np.sum(relative_allocations))
    denominator = n * np.sum(np.square(relative_allocations))
    return numerator / denominator


def plotDistribution(points, destination, lines=[], colors=[], labels=[]):
    plt.figure('dist')
    #plt.clf()

    num_hlines = len(lines)
    num_points = len(points)
    for i in range(num_hlines):
        plt.hlines(lines[i], 0.5, num_points+0.5, linestyles='-', label=labels[i], colors=colors[i])
    plt.bar(list(range(1, num_points+1)), points)

    # ax.set_title(r'\textbf{Flows:} ' + str(sendBehav) + r', \textbf{Capacity} ' + str(linkCap) + r'Mbps' + r', \textbf{Buffering}: ' + str(buffer_factor) + " * BDP")
    # ax.hlines([total_throughput], starttime, endtime, linestyles='-.', label='Average Throughput', colors='blue')
    # ax.set_xlabel(r'Time (s)')
    # ax.set_ylabel(r'Rate (Mbps)')
    plt.legend()

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(destination, dpi=150)
    plt.clf()
    plt.figure('overview')


#--------------------------------------------------------------------------------
# Plot congestion control data
def plotCwnd(figurename, ax, cwndData, ssthreshData, startTimestamp, endTimestamp, xticks, destination=None):
    plt.figure('overview')
    if destination is not None:
        plot, ax = plt.subplots()
    if len(PLOT_CWND_KEYS) == 0:
        return

    plotCwndData = {}
    print("start: ", startTimestamp)

    #print("cwndData: ", cwndData)
    for origin in PLOT_CWND_KEYS:
        ts = sorted(cwndData[origin].keys())

        hostID = re.match(r'.+\.(.+)$', origin).group(1)
        ts = [t for t in ts if float(t) >= startTimestamp and float(t) <= endTimestamp]
        displayTs = [float(t)-startTimestamp for t in ts]
        ax.plot(displayTs, [cwndData[origin][t] for t in ts], label=("h%s" % hostID), color=COLORS["h"+hostID][0], linewidth=0.5)

        # If ts == 0
        if displayTs == []:
            print("In plotCWnd: displayTs is empty. ts: ", ts)
            return

        plotCwndData[origin] = {}
        for i in range(len(ts)):
            plotCwndData[origin]['%.1f' % displayTs[i]] = cwndData[origin][ts[i]]

        ts = sorted(ssthreshData[origin].keys())
        if len(ts) != 0:
            ts = [t for t in ts if float(t) >= startTimestamp and float(t) <= endTimestamp]
            displayTs = [float(t)-startTimestamp for t in ts]
            ax.plot(displayTs, [ssthreshData[origin][t] for t in ts], ':', label=("h%s (ssthresh)" % hostID), color=COLORS["h"+hostID][0])

    with open(RESULT_FILE_PREFIX+'cwndData.json', 'w+') as pDF:
        json.dump(plotCwndData, pDF, sort_keys=True, indent=4)

    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'cwnd (Segments)')
    #ax.set_xticks(range(int(displayTs[-1]+1)))
    #ax.set_xticks(xticks)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    if destination is not None:
        plt.savefig(destination, dpi=300)
        plt.close(figurename)

# Plot Queue data
# Collect and print
def plotQueue(figurename, ax, ts, values, startTimestamp, endTimestamp, bwd_product, real_buffer_size, xticks, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()
    # Crop
    #print(ts - startTimestamp)
    #print(ts.shape)

    displayTs = [float(ts[i]) - startTimestamp + econfig['truncate_front'] for i in range(ts.shape[0]) if float(ts[i]) >= startTimestamp and float(ts[i]) <= endTimestamp]
    displayValues = [float(values[i])  for i in range(ts.shape[0]) if float(ts[i]) >= startTimestamp and float(ts[i]) <= endTimestamp]
    # Calculate average queue length
    avg_queue = np.average(displayValues)
    ax.plot(displayTs, displayValues, ':', label=("Switch 1"))

    utilizations = [min(v/bwd_product, 1.0) for v in displayValues]

    # Horizontal lines
    ax.plot([displayTs[0], displayTs[-1]], [avg_queue] * 2, '-', label='Avg Queue Size', color='blue', linewidth=0.5)
    ax.plot([displayTs[0], displayTs[-1]], [real_buffer_size] * 2, '-.', label='Buffer Size', color='green')
    ax.plot([displayTs[0], displayTs[-1]], [bwd_product] * 2, '-.', label='Bandwidth Delay Product', color='orange', linewidth=0.5)
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Queue Length (packets)')
    ax.set_ylim(bottom=0.0)
    ax.legend()
    if destination is not None:
        plt.savefig(destination, dpi=300)
        plt.close(figurename)

    avg_queue = np.average([max(dv - bwd_product, 0) for dv in displayValues])

    # Return stats that were calculated here
    return {'avg_queue': float(avg_queue), 'utilization_rel': np.mean(utilizations)}


def plotLoss(figurename, ax, tcpd_data, starttime, endtime, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()
    num_senders = econfig['inferred']['num_senders']
    linkCap = econfig['link_capacity']
    emulated_bufferCap = econfig['inferred']['bw_delay_product'] * float(econfig['switch_buffer'])
    real_bufferCap = econfig['inferred']['bw_delay_product'] + emulated_bufferCap
    sendBehav = econfig['inferred']['behavior_summary']
    sendBehav = sendBehav.replace('_', ' ')
    total_loss = 0
    for hostid in range(1, num_senders+1):
        label = "loss_" + str(hostid)
        if label in tcpd_data.columns:
            total_loss += np.sum([v for v in tcpd_data[label].values if not np.isnan(v)])
            ax.plot(tcpd_data.index.values, tcpd_data[label].values, ':', label=("h%s Loss" % (hostid)))

    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Loss')
    plt.locator_params(axis='both', nbins=20)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc=1)

    if destination is not None:
        plt.savefig(destination, dpi=300)
        plt.close(figurename)
    # Return stats that were calculated here
    if not total_loss >= 0:
        print("Packet sum is not a number! total_loss: ", total_loss)
        return {'avg_latency': 0, 'total_packets': 0}
    return {'total_loss': int(total_loss)}


def plotLatency(figurename, ax, tcpd_data, starttime, endtime, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()
    latency_sum = 0
    packet_sum = 0
    num_senders = econfig['inferred']['num_senders']
    print(tcpd_data.isnull().values.any())
    for hostid in range(1, num_senders+1):
        latency_label = "latency_sum_" + str(hostid)
        num_label = "num_" + str(hostid)
        if latency_label in tcpd_data.columns and num_label in tcpd_data.columns:
            latency_sum += np.sum([v for v in tcpd_data[latency_label].values if not np.isnan(v)])
            packet_sum += np.sum([v for v in tcpd_data[num_label].values if not np.isnan(v)])
            ax.plot(tcpd_data.index.values, tcpd_data[latency_label].values * (1000.0 / tcpd_data[num_label].values), ':',
                    label=("h%s Latency" % (hostid)))

    link_latency = econfig['link_latency']
    ax.hlines([link_latency], starttime, endtime, linestyles='-', label='Link Latency', colors='orange', linewidth=0.5)


    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Latency (ms)')
    plt.locator_params(axis='both', nbins=20)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc=1)

    if destination is not None:
        plt.savefig(destination, dpi=300)
        plt.close(figurename)

    # Return stats that were calculated here
    if not packet_sum >= 0:
        print("Packet sum is not a number! packet sum: ", packet_sum, "latency sum: ", latency_sum)
        return {'avg_latency': 0, 'total_packets': 0}
    return {'avg_latency': float(latency_sum * 1000.0 / packet_sum), 'total_packets': int(packet_sum)}

def plotJitter(figurename, ax, tcpd_data, starttime, endtime, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()

    num_senders = econfig['inferred']['num_senders']
    linkCap = econfig['link_capacity']
    emulated_bufferCap = econfig['inferred']['bw_delay_product'] * float(econfig['switch_buffer'])
    real_bufferCap = econfig['inferred']['bw_delay_product'] + emulated_bufferCap
    sendBehav = econfig['inferred']['behavior_summary']
    sendBehav = sendBehav.replace('_', ' ')

    jitter_sum = 0.0
    jitter_sum_squared = 0.0
    packet_sum = 0.0
    for hostid in range(1, num_senders+1):
        label_lat = "jitter_sum_" + str(hostid)
        label_sq_lat = "jitter_sum_sq_" + str(hostid)
        label_num = "num_" + str(hostid)
        if label_lat in tcpd_data.columns and label_num in tcpd_data.columns:
            jitter_sum += np.nansum(tcpd_data[label_lat].values)
            jitter_sum_squared += np.nansum(tcpd_data[label_sq_lat].values)
            packet_sum += np.sum([v for v in tcpd_data[label_num].values if not np.isnan(v)])
            index = tcpd_data.index.values
            values = tcpd_data[label_lat].values * (1000.0 / tcpd_data[label_num].values)
            ax.plot(index, values, ':', label=("h%s average PDV" % (hostid)))
            #ax.errorbar(index, values, stddev, fmt=':', label=("h%s Average Packet Delay Variation" % (hostid)))

    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Avg. Packet Delay Variation (ms)')
    plt.locator_params(axis='both', nbins=20)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc=1)

    if destination is not None:
        plt.savefig(destination, dpi=300)
        plt.close(figurename)
    # Return stats that were calculated here
    if not packet_sum >= 0:
        print("Packet sum is not an integer! packet sum: ", packet_sum, "jitter sum: ", jitter_sum)
        return {'avg_latency': 0, 'total_packets': 0}
    return {'avg_jitter': float(jitter_sum * 1000.0 / packet_sum), 'stdd_jitter': float(jitter_sum_squared / packet_sum)}



def plotMemory(figurename, ax, df, start, end, econfig, destination=None):
    plt.figure(figurename)
    if destination is not None:
        plot, ax = plt.subplots()
    ts = df[0]
    y_values = df[1]
    base_usage = y_values[0]
    y_values = [int(y_values[i])  for i in range(len(ts)) if float(ts[i]) >= start and float(ts[i]) <= end]
    ts = [float(ts[i]) - start + econfig['truncate_front'] for i in range(len(ts)) if float(ts[i]) >= start and float(ts[i]) <= end]
    y_values = y_values - base_usage
    ax.plot(ts, y_values, "-", label='System Memory Usage')
    average = np.average(y_values)
    if destination is not None:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(destination, dpi=300)
        plt.close(figurename)
    return {'avg_memory': average}

