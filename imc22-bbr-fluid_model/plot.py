#!/usr/bin/python3

import json
import yaml
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random

from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FuncFormatter


#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{amssymb}'

MSS = 1514

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def parse_model_result(result_file_name):

    with open(result_file_name, 'r') as data_file:
        json_data = json.load(data_file)

    if len(json_data) == 0:
        return None
    else:
        timestamp = result_file_name.split('/')[-1][:-5]
        for entry in json_data:
            entry['timestamp'] = timestamp
            if 'ufp' in entry.keys():
                ufp_names = ['rate_sensitivity', 'rate_exponent', 'latency_sensitivity',\
                             'inflation_sensitivity', 'loss_sensitivity']
                for i in range(len(ufp_names)):
                    entry[ufp_names[i]] = entry['ufp'][i]
                del entry['ufp']
        column_names = list(json_data[0].keys())

    data_frame = pd.DataFrame(json_data, columns=column_names)
    for r, _ in data_frame.iterrows():
        data_frame.at[r,'cc_combination'] = '/'.join(sorted(data_frame.at[r,'cc_combination']))
        data_frame.at[r,'source_latency_range'] = '-'.join([str(f) for f in data_frame.at[r,'source_latency_range']])
        bdp = math.ceil(data_frame.at[r, 'link_capacity'] * 1e6 / (MSS * 8 * 1000) * data_frame.at[r, 'link_latency'])
        data_frame.at[r, 'avg_queue'] = data_frame.at[r, 'avg_queue'] / int(bdp * data_frame.at[r, 'switch_buffer']) * 100
        data_frame.at[r, 'loss'] = data_frame.at[r, 'loss'] * 100
        data_frame.at[r, 'utilization'] = min(data_frame.at[r, 'utilization'], 1.0) * 100

    print('Parsed '+result_file_name)

    return data_frame


def get_model_dataframe(config):

    result_folder = 'results/' + config['name'] + '/fluid_model/'
    if not os.path.exists(result_folder):
        print("No model results to plot! Ignoring...")
        return None

    data_frames = []
    for data_file_name in [f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))]:
        if data_file_name[-4:] == 'json':
            data_frames.append( parse_model_result(result_folder + data_file_name) )

    return pd.concat(data_frames)


def parse_experiment_result(result_dir_name):

    with open(result_dir_name+'/config.yaml', 'r') as exp_config_file:
        exp_config = yaml.safe_load(exp_config_file)

    timestamp = result_dir_name.split('/')[-1]
    result_data = {
        'timestamp': timestamp,
        'senders': exp_config['senders'],
        'link_capacity': exp_config['link_capacity'], # to Gbps
        'switch_buffer': exp_config['switch_buffer'],
        'source_latency_range': '-'.join([str(f) for f in exp_config['source_latency_range']]),
        'qdisc': 'RED' if exp_config['use_red'] else 'Drop-tail',
    }

    cc_combination = exp_config['behavior_command'].replace('BBR2', 'BBRZ')
    cc_combination = ''.join([s for s in cc_combination if not (s.isdigit() or s == "-")])
    cc_combination = cc_combination.replace('BBRZ', 'BBR2')
    cc_combination = '/'.join(sorted(cc_combination.split('_')))
    result_data['cc_combination'] = cc_combination

    if 'utility_function_parameters' in exp_config.keys():
        ufp = exp_config['utility_function_parameters']
        result_data['rate_sensitivity']      = ufp[0]
        result_data['rate_exponent']         = ufp[1]
        result_data['latency_sensitivity']   = ufp[2]
        result_data['inflation_sensitivity'] = ufp[3]
        result_data['loss_sensitivity']      = ufp[4]

    with open(result_dir_name+'/stats.json', 'r') as data_file:
        json_data = json.load(data_file)

    json_data['loss'] = json_data['total_loss'] / (json_data['total_packets']+json_data['total_loss']) * 100 if json_data['total_packets'] != 0 else 0
    #json_data['loss'] = json_data['total_loss'] / json_data['total_packets'] * 100 if json_data['total_packets'] != 0 else 0
    json_data['utilization'] = min(json_data['utilization_rel'], 1) * 100
    bdp = exp_config['inferred']['bw_delay_product']
    json_data['avg_queue'] = json_data['avg_queue'] / (bdp * result_data['switch_buffer']) * 100

    result_data = {**result_data, **json_data}

    if len(result_data) == 0:
        return None
    else:
        column_names = list(result_data.keys())

    data_frame = pd.DataFrame([result_data], columns=column_names)

    print('Parsed '+result_dir_name + '/stats.json')

    return data_frame



def get_experiments_dataframe(config, result_folder=None):

    if result_folder is None:
        result_base_dir = 'results/' + config['name'] + '/mininet_experiments/'
    else:
        result_base_dir = result_folder

    result_dirs = []
    for dir_name, contained_dirs, contained_files in os.walk(result_base_dir):
        if 'stats.json' in contained_files:
            result_dirs.append(dir_name)

    if len(result_dirs) == 0:
        print("No experiment results to plot! Ignoring...")
        return None

    data_frames = []
    for result_dir in result_dirs:
        data_frames.append( parse_experiment_result(result_dir) )

    return pd.concat(data_frames)


line_map = {}

# key_to_label_map = {
#     'BBR': 'BBRv1', 
#     'BBRrandom': r'$\text{BBRv1}^{\ast}$',
#     'RENO': 'RENO',
#     'CUBIC': 'CUBIC',
#     'RENO/CUBIC': 'RENO/CUBIC',
#     'RENO/CUBICfriendly': r'$\text{RENO/CUBIC}^{\ast}$',
#     'BBR/RENO': 'BBRv1/RENO',
#     'BBR/CUBIC': 'BBRv1/CUBIC',
#     'RENOlazy': r'$\text{RENO}^{\sim}$',
#     'BBR2': 'BBRv2',
#     'BBR/BBR2': 'BBRv1/BBRv2',
#     'BBR2/RENO': 'BBRv2/RENO',
#     'BBR2/CUBIC': 'BBRv2/CUBIC',
#     'STABLE': 'CBR',
#     'VEGAS': r'VEGAS',
#     'PCC': r'PCC',
#     'PCC/RENO': r'PCC/RENO',
#     'PCC/VEGAS': r'PCC/VEGAS',
#     'BBR/PCC': r'PCC/BBR',
#     'PCCRENO': r'$\text{PCC}_{\text{RENO}}$',
#     'PCCVEGAS': r'$\text{PCC}_{\text{VEGAS}}$',
#     'PCCBBR': r'$\text{PCC}_{\text{BBR}}$',
#     'PCC/PCCRENO': r'$\text{PCC}$/$\text{PCC}_{\text{RENO}}$',
#     'PCC/PCCVEGAS': r'$\text{PCC}$/$\text{PCC}_{\text{VEGAS}}$',
#     'PCC/PCCBBR': r'$\text{PCC}$/$\text{PCC}_{\text{BBR}}$',
#     'PCCRENO/RENO': r'$\text{PCC}_{\text{RENO}}$/$\text{RENO}$',
#     'PCCVEGAS/VEGAS': r'$\text{PCC}_{\text{VEGAS}}$/$\text{VEGAS}$',
#     'BBR/PCCBBR': r'$\text{BBR}$/$\text{PCC}_{\text{BBR}}$',
#     'PCCFLEX': r'$\text{PCC}_{\text{FLEX}}$'
# }

key_to_label_map = {
    'BBR': 'BBRv1', 
    'BBRrandom': 'BBRv1*',
    'RENO': 'RENO',
    'CUBIC': 'CUBIC',
    'RENO/CUBIC': 'RENO/CUBIC',
    'RENO/CUBICfriendly': 'RENO/CUBIC*',
    'BBR/RENO': 'BBRv1/RENO',
    'BBR/CUBIC': 'BBRv1/CUBIC',
    'RENOlazy': 'RENO~',
    'BBR2': 'BBRv2',
    'BBR/BBR2': 'BBRv1/BBRv2',
    'BBR2/RENO': 'BBRv2/RENO',
    'BBR2/CUBIC': 'BBRv2/CUBIC',
    'STABLE': 'CBR',
    'VEGAS': 'VEGAS',
    'PCC': 'PCC',
    'PCC/RENO': 'PCC/RENO',
    'PCC/VEGAS': 'PCC/VEGAS',
    'BBR/PCC': 'PCC/BBR',
    'PCCRENO': 'PCC_RENO',
    'PCCVEGAS': 'PCC_VEGAS',
    'PCCBBR': 'PCC_BBR',
    'PCC/PCCRENO': 'PCC/PCC_RENO',
    'PCC/PCCVEGAS': 'PCC/PCC_VEGAS',
    'PCC/PCCBBR': 'PCC/PCC_BBR',
    'PCCRENO/RENO': 'PCC_RENO/RENO',
    'PCCVEGAS/VEGAS': 'PCC_VEGAS/VEGAS',
    'BBR/PCCBBR': 'BBR/PCC_BBR',
    'PCCFLEX': 'PCC_FLEX'
}



label_to_color_map = {
    'BBR':        (0.36, 0.54, 0.66),
    'BBRrandom':  (0.0, 1.0, 1.0),
    'RENO':       (0.53, 0.66, 0.42),
    'CUBIC':      (1.0, 0.44, 0.37),
    'RENO/CUBIC': (0.71, 0.65, 0.26),
    'RENO/CUBICfriendly': (0.59, 0.29, 0.0),
    'RENO/BBR':   (0.74, 0.2, 0.64),
    'CUBIC/BBR':  (0.44, 0.16, 0.39),
    'RENOlazy':   (0.16, 0.44, 0.39),
    'VEGAS':      (0.89, 0.61, 0.06),
    'PCC':        (0.52, 0.39, 0.44),
    'PCCFLEX':    (0.52, 0.39, 0.44),
    'PCCRENO':    (0.91, 0.17, 0.31),
    'PCCVEGAS':   (1.00, 0.75, 0.00),
    'PCCBBR':     (0.56, 0.74, 0.86),
    'PCC/RENO':   (0.53, 0.66, 0.42),
    'PCC/VEGAS':  (0.89, 0.61, 0.06),
    'BBR/PCC':    (0.36, 0.54, 0.66),
    'PCC/PCCRENO':  (0.91, 0.17, 0.31),
    'PCC/PCCVEGAS': (1.00, 0.75, 0.00),
    'PCC/PCCBBR':   (0.56, 0.74, 0.86),
    'PCCRENO/RENO':   (0.91, 0.17, 0.31), 
    'PCCVEGAS/VEGAS': (1.00, 0.75, 0.00),
    'BBR/PCCBBR':     (0.56, 0.74, 0.86)
}

label_to_marker_map = {
    'BBR':        'D',
    'BBRrandom':  'd',
    'RENO':       'o',
    'CUBIC':      's',
    'RENO/CUBIC': 'h',
    'RENO/CUBICfriendly': 'H',
    'BBR/RENO':   'P',
    'BBR/CUBIC':  'X',
    'RENOlazy':   '1',
    'BBR2':       'h',
    'BBR/BBR2':   'd',
    'BBR2/RENO':  'o',
    'BBR2/CUBIC': 's',
    'STABLE':     'H',
    'PCC':        '1',
    'VEGAS':      '|',
    'PCCFLEX':    '2',
    'PCCRENO':    '3',
    'PCCVEGAS':   '4',
    'PCCBBR':     'h',
    'PCC/PCCRENO':  '3',
    'PCC/PCCVEGAS': '4',
    'PCC/PCCBBR':   'h',
    'PCC/RENO':   'o',
    'PCC/VEGAS':  '|',
    'BBR/PCC':    'D',
    'PCCRENO/RENO': 'o',
    'PCCVEGAS/VEGAS': '|',
    'BBR/PCCBBR':     'D'
} 

def custom_formatter(x, pos):
    if x.is_integer():
        return f"{int(x):d}"
    else:
        return f"{x:.1f}"

def plot_metric(plot_handle, data_source, data_frame, x_param, y_param, z_param, other_params, plot_config):

    #global line_map
    #line_map = {}

    x_param_values = sorted(data_frame[x_param].unique())
    try:
        x_param_values = [x for x in x_param_values if x not in plot_config['dropped_keys']['x']]
    except KeyError:
        pass
    
    try:
        z_param_values = sorted(plot_config['selected_keys']['z'])
    except KeyError:
        z_param_values = sorted(data_frame[z_param].unique())

    sub_data_frame = data_frame
    for other_param_name, other_param_val in other_params:
        if other_param_name == 'source_latency_range':
            other_param_val = '-'.join([str(f) for f in other_param_val])
        sub_data_frame = sub_data_frame[sub_data_frame[other_param_name] == other_param_val]

    sub_data_frame = sub_data_frame.sort_values('timestamp', ascending=False)

    z_lines = {}
    available_x_for_z = {}
    for z_val in z_param_values:

        z_data = []
        z_val_frame = sub_data_frame[sub_data_frame[z_param] == z_val]

        data_complete = True
        available_x_param_values = []
        for x_val in x_param_values:
            x_val_frame = z_val_frame[z_val_frame[x_param] == x_val]

            y_param_vals = []
            for _, row in x_val_frame.iterrows():
                y_param_vals.append( row[y_param] )

            if len(y_param_vals) > 0:
                if data_source == 'Model': # Take chronologically last one
                    z_data.append( (y_param_vals[0], -1) )
                else: # Take average
                    '''
                    # Outlier removal
                    mean = np.mean(y_param_vals)
                    std  = np.std(y_param_vals)
                    pruned_y_param_vals = [y_val for y_val in y_param_vals if np.abs(y_val - mean) < std]
                    if len(pruned_y_param_vals) > 1:
                        y_param_vals = pruned_y_param_vals
                    '''
                    z_data.append( (np.mean(y_param_vals), np.std(y_param_vals)) )
                available_x_param_values.append(x_val)

        if len(available_x_param_values) > 0:
            z_lines[z_val] = z_data
            available_x_for_z[z_val] = available_x_param_values

    for z_val in z_lines.keys():
        z_marker = 'o'
        z_color  = (random.random(), random.random(), random.random())
        if z_val in label_to_marker_map.keys():
            z_marker = label_to_marker_map[z_val]
        if z_val in label_to_color_map.keys():
            z_color = label_to_color_map[z_val]
        if data_source == 'Model':
            line_map[z_val] = plot_handle.plot(available_x_for_z[z_val], \
                                               [z_lines[z_val][i][0] for i in range(len(available_x_for_z[z_val]))], \
                                               linestyle='-', marker=z_marker, \
                                               color=z_color, mfc='none', linewidth=0.8)[0]
        else:
            line_map[z_val] = plot_handle.errorbar(available_x_for_z[z_val], \
                                                   [z_lines[z_val][i][0] for i in range(len(available_x_for_z[z_val]))], \
                                                   linestyle='-', marker=z_marker,
                                                   yerr=[z_lines[z_val][i][1] for i in range(len(available_x_for_z[z_val]))],\
                                                   color=z_color, mfc='none', linewidth=0.8, capsize=5)[0]

    # x_param_label_map = {
    #     'switch_buffer': r'Buffer size [BDP]',
    #     'link_latency': 'Link latency [ms]',
    #     'senders': 'Number of senders',
    # }
    
    x_param_label_map = {
    'switch_buffer': 'Buffer size [BDP]',
    'link_latency': 'Link latency [ms]',
    'senders': 'Number of senders',
}
    
    if x_param in x_param_label_map.keys():
        x_param_label = x_param_label_map[x_param]
    else:
        x_param_label = r'' + x_param.replace('_', ' ')
    plot_handle.set_xlabel(x_param_label)
    plot_handle.set_xticks(x_param_values)
    plot_handle.set_xticklabels(["%.1f" % x if ((x/0.5)%2==1) else int(x) for x in  x_param_values])
    plot_handle.xaxis.set_label_coords(0.5, -0.2)
    
    x_param_values = sorted(data_frame[x_param].unique())

    # # Setting the x-axis to log2 scale
    # plot_handle.set_xscale('log', base=2)
    # plot_handle.set_xticks(x_param_values)  # Set ticks directly on the values
    # plot_handle.set_xticklabels(x_param_values)  # Set tick labels di
    
    
    plot_handle.set_xscale('log', base=2)  # Set logarithmic scale
    plot_handle.set_xticks(([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]))  # Define custom ticks
    plot_handle.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    
    #plot_handle.set_xscale('log', base=2)
    #plot_handle.set_xticks([1, 2, 4, 8, 16, 32, 64])
    #plot_handle.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    
    
    
    
    ## Set x-axis to logarithmic scale (log base 2)
    #plot_handle.set_xscale('log', base=2)
    #plot_handle.xaxis.set_major_locator(LogLocator(base=2))

    # y_param_label_map = {
    #     'avg_jitter': 'Jitter [ms]',
    #     'avg_queue': 'Buffer occupancy [\%]',
    #     'jain_fairness_index': 'Jain Fairness',
    #     'loss': 'Loss [\%]',
    #     'utilization': 'Utilization [\%]'
    # }
    
    y_param_label_map = {
    'avg_jitter': 'Jitter [ms]',
    'avg_queue': 'Buffer occupancy [%]',
    'jain_fairness_index': 'Jain Fairness',
    'loss': 'Loss [%]',
    'utilization': 'Utilization [%]'
}
    
    y_param_label = y_param_label_map[y_param]
    #plot_handle.set_ylabel(y_param_label.replace('_', '\_'))
    plot_handle.set_ylabel(y_param_label.replace('_', ' '))
    plot_handle.yaxis.set_label_coords(-0.16, 0.5)

    if 'focus' not in plot_config.keys():
        plot_handle.set_ylim(bottom=0)
    if 'y_limit' in plot_config.keys():
        plot_handle.set_ylim(top=plot_config['y_limit'])

    plot_handle.grid(which='major', axis='both', color='#DDDDDD')
    
    
    
    #title = r'\textbf{'+data_source+r'} '
    title = data_source
    plot_handle.set_title(title, pad=1, fontsize=10)
    plot_handle.legend(loc='upper left')



def generate_analysis_plots(config_name):

    global line_map
    
    

    with open(config_name, 'r') as config_file:
        config = yaml.safe_load(config_file)

    result_dir = 'results/' + config['name'] + '/'
    if not os.path.exists(result_dir):
        print("No results to plot!")
        sys.exit(1)

    plot_result_dir = result_dir + 'plots/'
    if not os.path.exists(plot_result_dir):
        os.mkdir(plot_result_dir)

    plots_config = config['plots']

    model_dataframe = None
    experiment_dataframe = None

    if len(list(plots_config)) == 0:
        print("No plots defined! Exiting...")
        sys.exit(0)

    if len([True for plot in plots_config.values() if plot['model_results']]) > 0:
        model_dataframe = get_model_dataframe(config)

    if len([True for plot in plots_config.values() if plot['experiment_results']]) > 0:
        experiment_dataframe = get_experiments_dataframe(config)

    for plot_name in plots_config.keys():
        print(plot_name)
        plot_config = plots_config[plot_name]
        print(plot_config['legend'])
        plot_model_results = (plot_config['model_results'] and model_dataframe is not None)
        plot_experiment_results = (plot_config['experiment_results'] and experiment_dataframe is not None)
        n_plots = (1 if plot_model_results else 0) + (1 if plot_experiment_results else 0)
        if n_plots == 0:
            continue

        
        if n_plots == 2:
            fig_width = 8 if plot_config['legend'] else 6
        else:
            fig_width = 5 if plot_config['legend'] else 3  
        fig, ax = plt.subplots(nrows=1, ncols=n_plots, figsize=(fig_width, 1.75))

        x_param = plot_config['x']
        y_param = plot_config['y']
        z_param = plot_config['z']
        other_params = []
        for other_param_name in plot_config['other'].keys():
            other_params.append( (other_param_name, plot_config['other'][other_param_name]) )

        min_ylim = 100
        max_ylim = 0
        plot_handles = []
        if plot_model_results:
            plot_handle = ax[0] if n_plots == 2 else ax
            plot_handles.append(plot_handle)
            plot_metric(plot_handle, 'Model', model_dataframe, x_param, y_param, z_param, other_params, plot_config)
            max_ylim = max(max_ylim, plot_handle.get_ylim()[1])
            min_ylim = min(min_ylim, plot_handle.get_ylim()[0])
            if n_plots == 1:
                plot_handle.yaxis.set_label_coords(-0.18, 0.5)
        if plot_experiment_results:
            plot_handle = ax[1] if n_plots == 2 else ax  
            plot_handles.append(plot_handle)
            plot_metric(plot_handle, 'Experiment', experiment_dataframe, x_param, y_param, z_param, other_params, plot_config)
            if n_plots == 2:
                plot_handle.set_yticklabels([])
                plot_handle.set_ylabel('')
                plot_handle.tick_params(axis='y', which='both', left=False, right=False) 
            max_ylim = max(max_ylim, plot_handle.get_ylim()[1])
            min_ylim = min(min_ylim, plot_handle.get_ylim()[0])
            if n_plots == 1:
                plot_handle.yaxis.set_label_coords(-0.18, 0.5)
        for plot_handle in plot_handles:
            plot_handle.set_ylim(bottom=min_ylim, top=max_ylim)

        plt.tight_layout()

        if n_plots == 2:
            if plot_config['legend']:
                subplot_limits = [0.215, 0.9, 0.08, 0.71]
            else:
                subplot_limits = [0.215, 0.9, 0.105, 0.995]
        else:
            if plot_config['legend']:
                subplot_limits = [0.215, 0.9, 0.13, 0.565]
            else:
                subplot_limits = [0.215, 0.9, 0.22, 0.995]
        plt.subplots_adjust(bottom=subplot_limits[0], top=subplot_limits[1], left=subplot_limits[2], right=subplot_limits[3], wspace=0)

        if plot_config['legend']:
            for label in line_map.keys():
                print("Label:", key_to_label_map.get(label, label)) 
                #if label in key_to_label_map.keys():
                    #line_map[label].set_label(key_to_label_map[label])
                line_map[label].set_label(key_to_label_map.get(label, str(label)))
                #line.set_label(key_to_label_map.get(label, label))
                #else:
                #    line_map[label].set_label(r''+str(label))
            #plt.figlegend(loc='best')
            #ax.legend(loc='best') 
            #plot_handle.legend(loc='right')
            plot_handle.legend(loc='upper left', bbox_to_anchor=(1.02, 1.15))


        plt.savefig(plot_result_dir+plot_name+'.pdf')
        plt.close()

        line_map = {}



if __name__ == '__main__':
    if len(sys.argv) > 1:
        generate_analysis_plots(sys.argv[1])
    else:
        print('Please provide a configuration.')