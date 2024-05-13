#!/usr/bin/python3

import sys
import yaml
import itertools
import os
import subprocess

from mininet_experiments.ccexperiment import main as ccmain


def run_cc(config, exp_dir, workpath=""):
    #f = open(expname + "outstream.txt")
    runname = config['name']
    print("Mininet Cleanup Start.")
    subprocess.run(["sudo", "mn",  "-c"],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("For Experiment: ", exp_dir, " doing run: " + runname)
    config['base_res_dir'] = exp_dir

    # Run the experiment
    ccmain(config)

    resultfolderpath = config['result_dir']

    if config['delete_raw_dump']:
        os.system('find ' + resultfolderpath +  '* -name "*.log" -print | grep hostdata | xargs sudo rm')


def get_config(config_name):

    with open(config_name, 'r') as config_file:
        config = yaml.safe_load(config_file)

    fixed_parameters = config['common_parameters']
    for pox_dir_candidate in config['experiment_parameters']['pox_directory']:
        if os.path.exists(pox_dir_candidate):
            fixed_parameters['pox_directory'] =  pox_dir_candidate
            break
    if 'pox_directory' not in fixed_parameters.keys():
        print("No valid pox_directory found! Aborting...")
        sys.exit(1)

    if 'delete_raw_dump' in config['experiment_parameters'].keys():
        fixed_parameters['delete_raw_dump'] = config['experiment_parameters']['delete_raw_dump']

    param_names = list(config['varying_parameters'].keys())
    config['param_combinations'] = []
    pc_counter = 0
    for param_comb in itertools.product(*[config['varying_parameters'][pn] for pn in param_names]):
        pc_map = fixed_parameters.copy()
        for i in range(len(param_names)):
            param_name = param_names[i]
            if param_name == 'cc_combination':
                N = param_comb[param_names.index('senders')]
                if len(param_comb[i]) == N:
                    cc_map = {}
                    for cc_name in param_comb[i]:
                        try:
                            cc_map[cc_name] += 1
                        except KeyError:
                            cc_map[cc_name] = 1
                    pc_map['behavior_command'] = '_'.join([ cc_name+'-'+str(cc_map[cc_name]) for cc_name in sorted(cc_map.keys()) ])
                else:
                    n_adopters_each_protocol = int(N/len(param_comb[i]))
                    pc_map['behavior_command'] = '_'.join([ cc+'-'+str(n_adopters_each_protocol) for cc in param_comb[i] ])
            elif param_name == 'qdisc':
                pc_map['use_red'] = True if param_comb[i] == 'RED' else False
            else:
                pc_map[param_names[i]] = param_comb[i]
        #pc_map['name'] = '_'.join([pc_map_key+':'+str(pc_map[pc_map_key]).replace(' ', '') for pc_map_key in pc_map.keys() if pc_map_key != 'pox_directory'])
        pc_map['name'] = str(pc_counter)
        pc_counter += 1
        config['param_combinations'].append(pc_map)

    return config


def main(config_name):

    config = get_config(config_name)

    result_folder = 'results/'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        os.chown(result_folder, os.stat('.').st_uid, os.stat('.').st_gid)
    result_folder += config['name'] + '/'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        os.chown(result_folder, os.stat('.').st_uid, os.stat('.').st_gid)
    result_folder += 'mininet_experiments/'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        os.chown(result_folder, os.stat('.').st_uid, os.stat('.').st_gid)

    with open(config['experiment_parameters']['base_config'], 'r') as base_config_file:
        base_config = yaml.safe_load(base_config_file)

    for pc_map in config['param_combinations']:
        exp_config = base_config.copy()
        exp_config.update(pc_map)
        print(pc_map)
        for i in range(config['experiment_parameters']['runs']):
            print("Run", i+1, "/", config['experiment_parameters']['runs'])
            run_cc(exp_config, result_folder)
            for dir_name, _, file_names in os.walk(result_folder):
                for file_name in file_names:
                    os.chown(os.path.join(dir_name, file_name), os.stat('.').st_uid, os.stat('.').st_gid)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a multi-experiment config file.")