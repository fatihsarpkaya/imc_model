#!/usr/bin/python3

import json
import yaml
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import re
import random

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{amssymb}'

MSS = 1514

def parse_model_result(result_file_path, file_name):

    with open(result_file_path, 'r') as data_file:
        json_data = json.load(data_file)
    json_data['timestamp'] = file_name

    if len(json_data) == 0:
        return None

    column_names = list(json_data.keys())
    data_frame = pd.DataFrame([json_data], columns=column_names)

    for r, _ in data_frame.iterrows():

        cc_combination_cardinality = len(data_frame.at[r, 'cc_combination'])
        cc_combination_expanded = []
        for cc in data_frame.at[r, 'cc_combination']:
            cc_combination_expanded += [cc] * int(data_frame.at[r, 'senders']/cc_combination_cardinality)

        data_frame.at[r, 'cc_combination'] = '/'.join(sorted(cc_combination_expanded))

        slr = data_frame.at[r, 'source_latency_range']
        slr_length = slr[1] - slr[0]
        net_latencies = []
        t_range = [ i*0.001 for i in range(len(data_frame.at[r, 'y'])) ]
        random.seed(1)
        for i in range(data_frame.at[r, 'senders']):
            net_latency = slr[0] + random.random()*slr_length + data_frame.at[r, 'link_latency']
            net_latencies.append(net_latency)
            bw_delay_product = 2*math.ceil(data_frame.at[r, 'link_capacity'] * 1e6 / (MSS * 8 * 1000) * (net_latency))
            data_frame.at[r, 'w_'+str(i)] = [d/bw_delay_product*100 for d in data_frame.at[r, 'w_'+str(i)]]
            if 'wmax_'+str(i) in data_frame.columns:
                data_frame.at[r, 'wmax_'+str(i)] = [d/bw_delay_product*100 for d in data_frame.at[r, 'wmax_'+str(i)]]
            if 'v_'+str(i) in data_frame.columns:
                data_frame.at[r, 'v_'+str(i)] = [d/bw_delay_product*100 for d in data_frame.at[r, 'v_'+str(i)]]
            if 'whi_'+str(i) in data_frame.columns:
                data_frame.at[r, 'whi_'+str(i)] = [d/bw_delay_product*100 for d in data_frame.at[r, 'whi_'+str(i)]]
                data_frame.at[r, 'wlo_'+str(i)] = [d/bw_delay_product*100 for d in data_frame.at[r, 'wlo_'+str(i)]]
        data_frame.at[r, 'source_latency_range'] = '-'.join([str(f) for f in data_frame.at[r,'source_latency_range']])
        data_frame.at[r, 'tau'] = [ max((d/(2*net_latencies[0])-1)*100, 0) for d in data_frame.at[r, 'tau']]

        N = data_frame.at[r, 'senders']
        for indicator_key in ['y', 'q', 'p', 'tau']:
            data_frame.at[r, indicator_key] = [t_range, data_frame.at[r, indicator_key]]

        for i in range(N):
            data_frame.at[r, 'w_'+str(i)] = [t_range, data_frame.at[r, 'w_'+str(i)]]
            data_frame.at[r, 'x_'+str(i)] = [t_range, data_frame.at[r, 'x_'+str(i)]]
            if 's_'+str(i) in data_frame.columns:
                data_frame.at[r, 's_'+str(i)] = [t_range, data_frame.at[r, 's_'+str(i)]]
            if 'wmax_'+str(i) in data_frame.columns:
                data_frame.at[r, 'wmax_'+str(i)] = [t_range, data_frame.at[r, 'wmax_'+str(i)]]
            if 'xbtl_'+str(i) in data_frame.columns:
                data_frame.at[r, 'xbtl_'+str(i)] = [t_range, data_frame.at[r, 'xbtl_'+str(i)]]
            if 'xmax_'+str(i) in data_frame.columns:
                data_frame.at[r, 'xmax_'+str(i)] = [t_range, data_frame.at[r, 'xmax_'+str(i)]]
            if 'xdel_'+str(i) in data_frame.columns:
                data_frame.at[r, 'xdel_'+str(i)] = [t_range, data_frame.at[r, 'xdel_'+str(i)]]
                data_frame.at[r, 'v_'+str(i)] = [t_range, data_frame.at[r, 'v_'+str(i)]]
                data_frame.at[r, 'tstr_'+str(i)] = [t_range, data_frame.at[r, 'tstr_'+str(i)]]
                data_frame.at[r, 'tmin_'+str(i)] = [t_range, data_frame.at[r, 'tmin_'+str(i)]]
                data_frame.at[r, 'tprt_'+str(i)] = [t_range, data_frame.at[r, 'tprt_'+str(i)]]
                data_frame.at[r, 'mprt_'+str(i)] = [t_range, data_frame.at[r, 'mprt_'+str(i)]]
            if 'whi_'+str(i) in data_frame.columns:
                data_frame.at[r, 'whi_'+str(i)] = [t_range, data_frame.at[r, 'whi_'+str(i)]]
                data_frame.at[r, 'wlo_'+str(i)] = [t_range, data_frame.at[r, 'wlo_'+str(i)]]
                data_frame.at[r, 'mdwn_'+str(i)] = [t_range, data_frame.at[r, 'mdwn_'+str(i)]]
                data_frame.at[r, 'mcrs_'+str(i)] = [t_range, data_frame.at[r, 'mcrs_'+str(i)]]
            if 'xpcg_'+str(i) in data_frame.columns:
                data_frame.at[r, 'xpcg_'+str(i)] = [t_range, data_frame.at[r, 'xpcg_'+str(i)]]

    print('Parsed '+result_file_path)

    return data_frame


def get_model_dataframe(config):

    result_folder = 'results/' + config['name'] + '/fluid_model/traces/'
    if not os.path.exists(result_folder):
        print("No model results to plot! Ignoring...")
        return None

    data_frames = []
    for data_file_name in [f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))]:
        if data_file_name[-4:] != 'json':
            continue
        data_frames.append( parse_model_result(result_folder + data_file_name, data_file_name[:-5]) )

    return pd.concat(data_frames)


def get_experiment_result(result_dir_name):

    with open(result_dir_name+'/config.yaml', 'r') as exp_config_file:
        exp_config = yaml.safe_load(exp_config_file)

    timestamp = result_dir_name.split('/')[-1]

    N = exp_config['senders']
    bw_delay_product = exp_config['inferred']['bw_delay_product']

    result_data = {
        'timestamp': timestamp,
        'senders': exp_config['senders'],
        'link_capacity': exp_config['link_capacity'], # to Gbps
        'switch_buffer': exp_config['switch_buffer'],
        'source_latency_range': '-'.join([str(f) for f in exp_config['source_latency_range']]),
        'qdisc': 'RED' if exp_config['use_red'] else 'Drop-tail',
        'link_latency': exp_config['link_latency']
    }

    cc_combination = exp_config['behavior_command'].replace('BBR2', 'BBRZ')
    cc_combination = ''.join([s if s != "_" else "/" for s in cc_combination if not (s.isdigit() or s == "-")])
    cc_combination = cc_combination.replace('BBRZ', 'BBR2')
    cc_combination = '/'.join(sorted(cc_combination.split('/')))
    cc_combination_cardinality = len(cc_combination.split('/'))
    cc_combination_expanded = []
    for cc in cc_combination.split('/'):
        cc_combination_expanded += [cc] * int(exp_config['senders']/cc_combination_cardinality)
    result_data['cc_combination'] = '/'.join(cc_combination_expanded)

    result_data['result_dir_name'] = result_dir_name

    if len(result_data) == 0:
        return None
    else:
        column_names = list(result_data.keys())

    data_frame = pd.DataFrame([result_data], columns=column_names)

    print('Parsed '+result_dir_name)

    return data_frame



def parse_experiment_result(df, row_idx):

    result_dir_name = df.loc[row_idx, 'result_dir_name']

    print(result_dir_name)

    with open(result_dir_name+'/config.yaml', 'r') as exp_config_file:
        exp_config = yaml.safe_load(exp_config_file)

    N = exp_config['senders']
    bw_delay_product = exp_config['inferred']['bw_delay_product']

    CC = []
    for cca in exp_config['behavior_command'].split('_'):
        cca_num = cca.split('-')
        CC += [cca_num[0]] * int(cca_num[1])

    for i in range(N):
        key_to_label_map['w_'+str(i)] = CC[i]
        key_to_label_map['x_'+str(i)] = CC[i]

    slr = exp_config['source_latency_range']
    slr_length = slr[1] - slr[0]
    source_latencies = []
    net_latencies = []
    net_bw_delay_products = []
    random.seed(1)
    for i in range(N):
        source_latencies.append( float("%.1f" % random.uniform(slr[0], slr[1])) )
        net_latencies.append( exp_config['link_latency'] + source_latencies[i] )
        net_bw_delay_product = 2*bw_delay_product + 2 * exp_config['link_capacity'] * 1e6 / (MSS * 8 * 1000) * source_latencies[i]
        net_bw_delay_products.append( net_bw_delay_product )

    agg_data = {}

    # ---------------------------------------------------
    # Analyze TCPDumpData
    trace_data = pd.read_csv(result_dir_name+'/condensed/tcpd_dataframe.csv')
    start_timestamp = trace_data['abs_ts'].values[0]
    end_timestamp   = trace_data['abs_ts'].values[-1]
    timestep = exp_config['plot_load_resolution']
    trace_data_keys = trace_data.columns
    converter = 8.0 / (1e6 * exp_config['link_capacity']) * 100 # B to percent of link capacity
    xs = []
    xdels = []
    inflights = []
    #vcalcs = []

    for i in range(N):
        xs.append( [] )
        xdels.append( [] )
        inflights.append( [] )

    ys = []
    ps = []
    taus = []
    vcalc = 0
    timestamps = []
    for r, _ in trace_data.iterrows():
        rel_timestamp = trace_data.at[r,'timestamp']
        if rel_timestamp < exp_config['truncate_front'] or\
           rel_timestamp > exp_config['send_duration'] - exp_config['truncate_back']:
           continue
        #ys.append( trace_data.at[r, 'total_load'] * converter / timestep )
        x_sum = 0
        for i in range(N):
            xs[i].append( trace_data.at[r, 'load_sent_'+str(i+1)] * converter / timestep )
            if np.isnan(xs[i][-1]):
                xs[i][-1] = 0
            x_sum += xs[i][-1]
            xdels[i].append( trace_data.at[r, 'bytes_acked_'+str(i+1)] / 1448 * MSS * converter / timestep )
            if np.isnan(xdels[i][-1]):
                xdels[i][-1] = 0
            #vcalc = max(vcalc + xs[i][-1] - xdels[i][-1], 0)
            #vcalcs[i].append( vcalc )
            n_pkts = trace_data.at[r, 'num_'+str(i+1)]
            if n_pkts != 0:
                avg_inflight = trace_data.at[r, 'inflight_sum_'+str(i+1)] / trace_data.at[r, 'num_'+str(i+1)]
            else:
                avg_inflight = 0
            # Weird: outbound traffic is only measured after first link (but before buffer), 
            #        so we estimate inflight on that link from current rate
            #        (current rate is ok because first-link delay is less than aggregation timestep)
            additional_inflight =  source_latencies[i] * (trace_data.at[r, 'load_'+str(i+1)] / timestep / 1000)
            avg_inflight += additional_inflight
            inflights[i].append( (avg_inflight / 1448) / net_bw_delay_products[i] * 100 )
        rel_timestamp = float("%.2f" % (rel_timestamp - exp_config['truncate_front']))
        timestamps.append( rel_timestamp )
        ys.append( x_sum )
        n_pkts = 0
        n_losses = 0
        latencies = []
        for i in range(N):
            n_pkts += trace_data.at[r, 'num_'+str(i+1)]
            n_losses += trace_data.at[r, 'loss_'+str(i+1)]
            if trace_data.at[r, 'num_'+str(i+1)] != 0:
                latencies.append( trace_data.at[r, 'latency_sum_'+str(i+1)] / trace_data.at[r, 'num_'+str(i+1)] )
        if n_pkts != 0:
            ps.append( min(n_losses/(n_pkts+n_losses), 1.0)*100 )
        else:
            ps.append(0)
        mean_latency = np.mean(latencies) if len(latencies) > 0 else 0
        measured_latency = 2*source_latencies[0] + exp_config['link_latency'] + mean_latency * 1000
        taus.append( (measured_latency/(2*net_latencies[0])-1)*100 )

    agg_data['y'] = [ timestamps, ys ]
    agg_data['p'] = [ timestamps, ps ]
    agg_data['tau'] = [ timestamps, taus ]
    for i in range(N):
        agg_data['x_'+str(i)] = [ timestamps, xs[i] ]
        agg_data['xdel_'+str(i)] = [ timestamps, xdels[i] ]
        agg_data['v_'+str(i)] = [ timestamps, inflights[i] ]
        #agg_data['vcalc_'+str(i)] = [ timestamps, vcalcs[i] ]

    # ---------------------------------------------------
    # Analyze queue data
    queue_data = pd.read_csv(result_dir_name+'/queue_length.csv', names=['timestamp', 'queue_length', 'queue_length_2'])
    buffer_size = exp_config['inferred']['buffer_size']
    queue_sizes = {}
    timestep = exp_config['tc_queue_sample_period']
    for r, _ in queue_data.iterrows():
        queue_timestamp = queue_data.at[r, 'timestamp'] 
        if queue_timestamp > start_timestamp + exp_config['truncate_front'] and\
           queue_timestamp < start_timestamp + exp_config['send_duration'] - exp_config['truncate_back']:

            rel_timestamp = int((queue_timestamp - start_timestamp - exp_config['truncate_front'])/timestep) * timestep
            rel_timestamp = float( "%.2f" % rel_timestamp )

            corrected_bw_delay_product = bw_delay_product + 4
            queue_val = queue_data.at[r, 'queue_length']
            queue = max(queue_val - corrected_bw_delay_product, 0) / (buffer_size - corrected_bw_delay_product) * 100
            try:
                queue_sizes[rel_timestamp] += [queue]
            except KeyError:
                queue_sizes[rel_timestamp] = [queue]
    
    agg_data['q'] = [ list(queue_sizes.keys()), [np.mean(q) for q in list(queue_sizes.values())] ]

    # ---------------------------------------------------
    # Analyze hostlogs
    timestep = exp_config['cwind_sampling_period']
    for i in range(N):
        cwnds = {}
        xbtls = {}
        tmins = {}
        pacing_rates = {}
        delivery_rates = {}
        net_bw_delay_product = net_bw_delay_products[i]
        with open(result_dir_name+'/hostlogs/h'+str(i+1)+'.log') as host_log_file:
            line = host_log_file.readline()
            while line:
                m = re.match(r'^(\d+\.\d+).*cwnd:(\d+)$', line.strip())
                if m != None:
                    timestamp = float(m.group(1))
                    if timestamp > start_timestamp + exp_config['truncate_front'] and\
                       timestamp < start_timestamp + exp_config['send_duration'] - exp_config['truncate_back']:
                        cwnd = float(m.group(2))
                        rel_timestamp = int((timestamp - start_timestamp - exp_config['truncate_front'])/timestep)*timestep
                        rel_timestamp = float("%.2f" % rel_timestamp)
                        try:
                            cwnds[rel_timestamp] += [cwnd]
                        except KeyError:
                            cwnds[rel_timestamp] = [cwnd]

                    line = host_log_file.readline()
                    continue

                m = re.match(r'^(\d+\.\d+).*btl_bw (\S+).*mrtt (\S+).*pacing_rate (\S+).*delivery_rate (\S+)$', line.strip())
                if m != None:
                    timestamp = float(m.group(1))
                    if timestamp > start_timestamp + exp_config['truncate_front'] and\
                       timestamp < start_timestamp + exp_config['send_duration'] - exp_config['truncate_back']:
                        rel_timestamp = int((timestamp - start_timestamp - exp_config['truncate_front'])/timestep)*timestep
                        rel_timestamp = float("%.2f" % rel_timestamp)
                        xbtl = float(m.group(2)) / 1448 * 1514
                        tmin = float(m.group(3))
                        pacing_rate = float(m.group(4)) / 1448 * 1514
                        delivery_rate = float(m.group(5)) / 1448 * 1514
                        try:
                            xbtls[rel_timestamp] += [xbtl]
                            tmins[rel_timestamp] += [tmin]
                            pacing_rates[rel_timestamp] += [pacing_rate]
                            delivery_rates[rel_timestamp] += [delivery_rate]
                        except KeyError:
                            xbtls[rel_timestamp] = [xbtl]
                            tmins[rel_timestamp] = [tmin]
                            pacing_rates[rel_timestamp] = [pacing_rate]
                            delivery_rates[rel_timestamp] = [delivery_rate]

                line = host_log_file.readline()
        
        agg_data['w_'+str(i)] = [ list(cwnds.keys()), [np.mean(l)/(net_bw_delay_product+4)*100 for l in list(cwnds.values())] ]

        if 'BBR' in CC[i]:
            agg_data['xbtl_'+str(i)] = [ list(xbtls.keys()), [np.mean(l)/exp_config['link_capacity']*100 for l in list(xbtls.values())] ]
            agg_data['tmin_'+str(i)] = [ list(tmins.keys()), [(np.mean(l)-2*net_latencies[i])/(2*net_latencies[i])*100 for l in list(tmins.values())] ]
            agg_data['xpcg_'+str(i)] = [ list(pacing_rates.keys()), [np.mean(l)/exp_config['link_capacity']*100 for l in list(pacing_rates.values())] ]
            agg_data['xdel_'+str(i)] = [ list(delivery_rates.keys()), [np.mean(l)/exp_config['link_capacity']*100 for l in list(delivery_rates.values())] ]

    # Aggregate delivery rate
    ydels = {}
    for i in range(N):
        if 'BBR' not in CC[i]:
            continue
        for n in range(len(agg_data['xdel_'+str(i)][0])):
            timestamp = agg_data['xdel_'+str(i)][0][n]
            del_rate = agg_data['xdel_'+str(i)][1][n]
            try:
                ydels[timestamp] += [del_rate]
            except KeyError:
                ydels[timestamp]  = [del_rate]
    ydel_timestamps = sorted(list(ydels.keys()))
    agg_data['ydel'] = [ ydel_timestamps, [sum(ydels[timestamp])/len(ydels[timestamp])*N for timestamp in ydel_timestamps] ]  # Need extrapolation   


    # ---------------------------------------------------
    # Analyze BBR2 internals (if existing)
    bbr2_internals_filename = result_dir_name+'/condensed/bbr2_internals.csv'
    if os.path.exists(bbr2_internals_filename):
        wlos = []
        whis = []
        bdps = []
        for i in range(N):
            wlos.append( {} )
            whis.append( {} )
            bdps.append( {} )
    
            bbr2_internals_data = pd.read_csv(bbr2_internals_filename)
            for r, _ in bbr2_internals_data.iterrows():
                rel_timestamp = bbr2_internals_data.at[r,'timestamp']
                if rel_timestamp < exp_config['truncate_front'] or\
                   rel_timestamp > exp_config['send_duration'] - exp_config['truncate_back']:
                   continue
                rel_timestamp = float("%.2f" % (rel_timestamp - exp_config['truncate_front']))
                sender_id = int(bbr2_internals_data.at[r, 'sender_id']) - 1
                inflight_lo = int(bbr2_internals_data.at[r, 'inflight_lo'])
                inflight_hi = int(bbr2_internals_data.at[r, 'inflight_hi'])
                bdp = int(bbr2_internals_data.at[r, 'bdp'])
                if inflight_lo != -1:
                    try:
                        wlos[sender_id][rel_timestamp].append( inflight_lo )
                    except KeyError:
                        wlos[sender_id][rel_timestamp] = [ inflight_lo ]
                if inflight_hi != -1:
                    try:
                        whis[sender_id][rel_timestamp].append( inflight_hi )
                    except KeyError:
                        whis[sender_id][rel_timestamp] = [ inflight_hi ]
                try:
                    bdps[sender_id][rel_timestamp].append( bdp )
                except KeyError:
                    bdps[sender_id][rel_timestamp] = [ bdp ]
                
    
        for i in range(N):
            if CC[i] == 'BBR2':
                agg_data['wlo_'+str(i)] = [ list(wlos[i].keys()), [np.mean(l)/net_bw_delay_products[i]*100 for l in list(wlos[i].values()) ] ]
                agg_data['whi_'+str(i)] = [ list(whis[i].keys()), [np.mean(l)/net_bw_delay_products[i]*100 for l in list(whis[i].values()) ] ]
                agg_data['bdp_'+str(i)] = [ list(bdps[i].keys()), [np.mean(l)/net_bw_delay_products[i]*100 for l in list(bdps[i].values()) ] ]



    # ---------------------------------------------------
    # Analyze PCC internals (if existing)
    pcc_internals_filename = result_dir_name+'/condensed/pcc_internals.csv'
    if os.path.exists(pcc_internals_filename):
        utils = []
        for i in range(N):
            utils.append( {} )
    
        pcc_internals_data = pd.read_csv(pcc_internals_filename)
        for r, _ in pcc_internals_data.iterrows():
            rel_timestamp = pcc_internals_data.at[r,'timestamp']
            if rel_timestamp < exp_config['truncate_front'] or\
               rel_timestamp > exp_config['send_duration'] - exp_config['truncate_back']:
               continue
            rel_timestamp = float("%.2f" % (rel_timestamp - exp_config['truncate_front']))
            sender_id = int(pcc_internals_data.at[r, 'sender_id'])
            util = int(pcc_internals_data.at[r, 'util'])
            try:
                utils[sender_id][rel_timestamp].append( util )
            except KeyError:
                utils[sender_id][rel_timestamp] = [ util ]

        for i in range(N):
            if CC[i] == 'PCCFLEX':
                for timestamp in utils[i].keys():
                    utils[i][timestamp] = np.mean(utils[i][timestamp])
                util_vals = list(utils[i].values())
                if len(util_vals) == 0:
                    continue
                max_util = max(util_vals)
                min_util = min(util_vals)
                agg_data['util_'+str(i)] = [ list(utils[i].keys()), [(u-min_util)/(max_util-min_util)*100 for u in util_vals] ]

    for key in agg_data.keys():
        df.loc[row_idx, key] = json.dumps(agg_data[key])

    print('Parsed '+result_dir_name)



def get_experiments_dataframe(config):

    result_base_dir = 'results/' + config['name'] + '/mininet_experiments/'

    result_dirs = []
    for dir_name, contained_dirs, contained_files in os.walk(result_base_dir):
        if 'stats.json' in contained_files:
            result_dirs.append(dir_name)

    if len(result_dirs) == 0:
        print("No experiment results to plot! Ignoring...")
        return None

    data_frames = []
    for result_dir in result_dirs:
        data_frames.append( get_experiment_result(result_dir) )

    return pd.concat(data_frames)



key_to_label_map = {
    'y': 'Rate',
    'w_0': 'Cwnd',
    'w_1': 'Cwnd',
    'x_0': r'$x_1(t)$',
    'x_1': 'Rate of flow 2',
    'xbtl_0': r'$x_1^{\mathrm{btl}}(t)$',
    'xbtl_1': r'$x_2^{\mathrm{btl}}(t)$',
    'xdel_1': r'$x_2^{\mathrm{del}}(t)$',
    'xmax_0': r'$x_1^{\max}(t)$',
    'xmax_1': r'$x_1^{\max}(t)$',
    'q': 'Queue',
    'p': 'Loss',
    'tau': 'RTT'
}

line_map = {}

label_to_color_map = {
    'w_0':     (0.36, 0.54, 0.66), # Air force blue
    'whi_0':   (0.0, 1.0, 1.0),    # Aqua blue
    'v_0':     (0.0,  0.75, 1.0),  # Capri light Blue
    'x_0':     (1.0, 0.44, 0.37),  # Bittersweet red
    'x_1':     (1.0, 0.44, 0.37),  # Bittersweet red
    'y':       (1.0, 0.44, 0.37),  # Bittersweet red
    'xpcg_1':  (1.0, 0.44, 0.37),  # Bittersweet red
    'xpcg_0':  (1.0, 0.65, 0.79),  # Carnation pink
    'xdel_0':  (0.74, 0.2, 0.64),  # Byzantine purple
    'xbtl_0':  (0.87, 0.26, 0.51), # Blush dark rose
    'xmax_0':  (0.44, 0.16, 0.39), # Byzantium dark purple
    'q':       (0.0, 0.42, 0.24),  # Brass green
    'p':       (0.59, 0.29, 0.0),  # Brown
    'tau':     (1.0, 0.75, 0.0),    # Amber yello
    'util_0':  (0.6, 0.6, 0.6),
    'gmma_0':  (0.1, 0.8, 0.5)
}

key_to_linestyle_map = {
    'w_0':    (0, (2, 1)),
    'whi_0':  (0, (7, 3)),
    'v_0':    (0, (5, 1)),
    'x_0':    'solid',
    'x_1':    'solid',
    'xpcg_1': 'solid',
    'y':      'solid',
    'xpcg_0': (0, (1, 2)),
    'xdel_0': (0, (3, 1, 1, 1)),
    'xbtl_0': (0, (8, 1)),
    'xmax_0': (0, (1, 1)),
    'q':      (0, (3, 1, 1, 1)),    
    'p':      (0, (1, 1)),
    'tau':    (0, (3, 1, 1, 1, 1, 1)),
    'util_0': (0, (1, 1)),
    'gmma_0': (0, (2, 1))
}

def plot_trace(plot_handle, data_source, data_frame, metrics, smoothed_metrics, other_params, trace_length, focus=False):

    sub_data_frame = data_frame
    for other_param_name, other_param_val in other_params:
        if other_param_name == 'source_latency_range':
            other_param_val = '-'.join([str(f) for f in other_param_val])
        if other_param_name == 'cc_combination':
            n_senders = [opv for opn,opv in other_params if opn == 'senders'][0]
            protocols = other_param_val.split('/')
            n_protocols = len(protocols)
            n_senders_per_protocol = int(n_senders/n_protocols)
            protocols_expanded = []
            for proto in protocols:
                protocols_expanded += [proto] * n_senders_per_protocol
            other_param_val = '/'.join(protocols_expanded)
        sub_data_frame = sub_data_frame[sub_data_frame[other_param_name] == other_param_val]
        if sub_data_frame.empty:
            print(data_source, "Empty query result for:", other_params)
            return    

    sub_data_frame = sub_data_frame.sort_values('timestamp', ascending=False)
    sub_data_frame = sub_data_frame.head(1)

    if data_source == 'Experiment':
        parse_experiment_result(sub_data_frame, 0)

    data_row = (sub_data_frame.to_dict(orient='records'))[0]

    timestep = round( (trace_length / len(data_row[metrics[0]]))*1000 ) / 1000

    plot_max_val = 0.0

    for key in metrics:
        try:
            key_data = data_row[key]
        except KeyError:
            print(data_source, key, "does not exist")
            continue

        if data_source == 'Experiment':
            key_data = json.loads(key_data)

        print(key)
        t_range = key_data[0]
        t_range = [t for t in t_range if t < trace_length]
        vals = key_data[1][:len(t_range)]

        if data_source == 'Experiment' and key in smoothed_metrics:
            max_vals = []
            smoothed_vals = []
            SMOOTH_STEP = 2
            for j in range(len(vals)):
                smoothed_vals.append( np.mean(vals[max(0,j-SMOOTH_STEP):min(j+SMOOTH_STEP,len(vals))]) )
            vals = smoothed_vals
        color = label_to_color_map[key] if key in label_to_color_map.keys() else (random.random(), random.random(), random.random())
        linestyle = key_to_linestyle_map[key] if key in key_to_linestyle_map else (0, (1, 1))
        line_map[key] = plot_handle.plot(t_range, vals, linewidth=0.75, color=color, linestyle=linestyle)[0]
        plot_max_val = max(plot_max_val, max(vals))

    if trace_length % 3 != 0 and not focus:
        trace_length = math.ceil(trace_length/3)*3

    if not focus:
        plot_handle.set_xticks( [i*(trace_length/3) for i in range(4)] )
        plot_handle.set_xticklabels( [str(i*int(trace_length/3)) for i in range(4)] )

    plot_handle.grid(which='major', axis='both', color='#DDDDDD')

    if not focus:
        plot_handle.plot([0, trace_length], [100, 100], ':', color='k')

    plot_handle.set_xlim([-0.05*trace_length, 1.05*trace_length])
    plot_handle.set_ylim(bottom=0.0, top=max(1.1*plot_max_val, 110))

    plot_handle.yaxis.set_label_coords(-0.12, 0.5)
    plot_handle.xaxis.set_label_coords(0.5, -0.2)

    plot_handle.set_xlabel(r'Time [s]')

    title = r'\textbf{'+data_source+r'} '
    plot_handle.set_title(title, pad=1, fontsize=10)



def generate_analysis_plots(config_name):

    with open(config_name, 'r') as config_file:
        config = yaml.safe_load(config_file)

    result_dir = 'results/' + config['name'] + '/'
    if not os.path.exists(result_dir):
        print("No results to plot!")
        sys.exit(1)

    plot_result_dir = result_dir + 'plots/'
    if not os.path.exists(plot_result_dir):
        os.mkdir(plot_result_dir)

    plot_config = config['trace_plots']

    model_dataframe = None
    experiment_dataframe = None

    if len([True for plot in plot_config.values() if plot['model_results']]) > 0:
        model_dataframe = get_model_dataframe(config)

    experiment_dataframe = None

    if len([True for plot in plot_config.values() if plot['experiment_results']]) > 0:
        experiment_dataframe = get_experiments_dataframe(config)

    trace_length = config['common_parameters']['send_duration']\
                 - config['common_parameters']['truncate_front']\
                 - config['common_parameters']['truncate_back']

    for plot_name in plot_config.keys():
        print(plot_name)
        plot_model_results = (plot_config[plot_name]['model_results'] and model_dataframe is not None)
        plot_experiment_results = (plot_config[plot_name]['experiment_results'] and experiment_dataframe is not None)
        plot_model_results2 = ('model_results2' in plot_config[plot_name].keys() and plot_config[plot_name]['model_results2'] and model_dataframe is not None)
        n_plots = (1 if plot_model_results else 0) + (1 if plot_experiment_results or plot_model_results2 else 0)
        if n_plots == 0:
            continue

        if 'fig_width' in plot_config[plot_name].keys():
            fig_width = plot_config[plot_name]['fig_width']
            fig_height = plot_config[plot_name]['fig_height']
        else:
            fig_width = 2
            fig_height = 1.75
            if n_plots == 2:
                fig_width =4
                if plot_config[plot_name]['legend']:
                    if plot_config[plot_name]['legend_bottom']:
                        fig_height = 2
                    else:
                        fig_width = 5.5
            else:
                if plot_config[plot_name]['legend']:
                    if plot_config[plot_name]['legend_bottom']:
                        fig_height = 2
                    else:
                        fig_width = 3.5

        fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(fig_width, fig_height))


        other_params = []
        for other_param_name in plot_config[plot_name]['other'].keys():
            other_params.append( (other_param_name, plot_config[plot_name]['other'][other_param_name]) )

        if not plot_config[plot_name]['paper_version']:
            title = r''
            linelength = len(title)
            for op in other_params:
                title_addition = r'\textit{'+op[0].replace('_', '\_')+r'}: '+str(op[1]) + ", "
                linelength += len(title_addition)
                title += title_addition
                if linelength > 40:
                    title += '\n'
                    linelength = 0
            if title[-1] == '\n':
                title = title[:-3]
            else:
                title = title[:-2]
            plt.suptitle(title)

        min_ylim = 0
        max_ylim = 0
        plot_handles = []
        smoothed_metrics = plot_config[plot_name]['smoothed_metrics'] if 'smoothed_metrics' in plot_config[plot_name].keys() else []
        focus = plot_config[plot_name]['focus'] if 'focus' in plot_config[plot_name].keys() else False
        if plot_model_results:
            plot_handle = ax[0] if n_plots == 2 else ax
            plot_handles.append(plot_handle)
            plot_handle.set_ylabel(r'\%')
            plot_trace(plot_handle, 'Model', model_dataframe, plot_config[plot_name]['metrics'], smoothed_metrics, other_params, trace_length, focus)
            max_ylim = max(max_ylim, plot_handle.get_ylim()[1])
        if plot_experiment_results:
            plot_handle = ax[1] if n_plots == 2 else ax
            if n_plots == 2:
                plot_handle.set_yticklabels([])
                plot_handle.tick_params(axis='y', which='both', left=False, right=False)           
            plot_handles.append(plot_handle)
            plot_trace(plot_handle, 'Experiment', experiment_dataframe, plot_config[plot_name]['metrics'], smoothed_metrics, other_params, trace_length, focus)
            max_ylim = max(max_ylim, plot_handle.get_ylim()[1])
        elif plot_model_results2:
            plot_handle = ax[1] if n_plots == 2 else ax
            if n_plots == 2:
                plot_handle.set_yticklabels([])
                plot_handle.tick_params(axis='y', which='both', left=False, right=False)           
            plot_handles.append(plot_handle)
            plot_trace(plot_handle, 'Model', model_dataframe, plot_config[plot_name]['metrics2'], smoothed_metrics, other_params, trace_length, focus)
            max_ylim = max(max_ylim, plot_handle.get_ylim()[1])
        if 'y_limit' in plot_config[plot_name].keys():
            y_limit = plot_config[plot_name]['y_limit']
            if isinstance(y_limit, list):
                min_ylim = y_limit[0]
                max_ylim = y_limit[1]
            else:
                max_ylim = y_limit
        for plot_handle in plot_handles:
            plot_handle.set_ylim(bottom=min_ylim, top=max_ylim)

        plt.tight_layout()

        if n_plots == 2:
            if plot_config[plot_name]['legend']:
                if plot_config[plot_name]['legend_bottom']:
                    subplot_limits = [0.35, 0.9, 0.1, 0.995]
                else:
                    subplot_limits = [0.215, 0.9, 0.08, 0.81]
            else:
                subplot_limits = [0.215, 0.9, 0.105, 0.995]
        else:
            if plot_config[plot_name]['legend']:
                if plot_config[plot_name]['legend_bottom']:
                    subplot_limits = [0.35, 0.9, 0.17, 0.995]
                else:
                    subplot_limits = [0.215, 0.9, 0.13, 0.75]
            else:
                subplot_limits = [0.215, 0.9, 0.22, 0.995]
        plt.subplots_adjust(bottom=subplot_limits[0], top=subplot_limits[1], left=subplot_limits[2], right=subplot_limits[3], wspace=0)

        if plot_config[plot_name]['paper_version']:
            if plot_config[plot_name]['legend']:
                for label in line_map.keys():
                    if 'legend_keys' in plot_config[plot_name] and label in plot_config[plot_name]['legend_keys'].keys():
                        legend_key = plot_config[plot_name]['legend_keys'][label]
                    else:
                        label_components = label.split('_')
                        legend_key = r'$\mathit{'+label_components[0]+r'}'
                        if len(label_components) > 1:
                            legend_key += r'_{'+label_components[1]+r'}'
                        legend_key += r'$'
                    line_map[label].set_label(legend_key)
                if plot_config[plot_name]['legend_bottom']:
                    plt.figlegend(loc='lower center', ncol=5, handlelength=(n_plots*1.0), columnspacing=(n_plots*1.0))
                else:
                    plt.figlegend(loc='right', handlelength=(n_plots*1.0))

        plt.savefig(plot_result_dir+plot_name+'.pdf')
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        generate_analysis_plots(sys.argv[1])
    else:
        print('Please provide a configuration.')