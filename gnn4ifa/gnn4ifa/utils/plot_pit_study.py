# from gnn4ifa.data import IfaDataset
#
# dataset = IfaDataset(root='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
#                      download_folder='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data')
# print('dataset.raw_dir: {}'.format(dataset.raw_dir))
# print('dataset.processed_dir: {}'.format(dataset.processed_dir))
import os
import pandas as pd
import numpy as np
import glob
import pickle as pkl
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import random

SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIG_SIZE = 40
plt.rc('font', size=BIG_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title
# font = {'size': 25}
# matplotlib.rc('font', **font)
warnings.filterwarnings("ignore")


def plot_ifa_effects(download_folder, scenarios, topologies):
    # Check if pickle files are available for the pit sizes and satisfaction rates
    out_path = os.path.join(os.getcwd(), '..', 'output', 'pickles', 'ifa_effects')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        file_name = os.path.join(out_path, 'avg_std_pit_sizes_for_time.pkl')
        with open(file_name, 'rb') as handle:
            pit_sizes = pkl.load(handle)
        file_name = os.path.join(out_path, 'avg_std_satisfaction_rates_for_time.pkl')
        with open(file_name, 'rb') as handle:
            satisfaction_rates = pkl.load(handle)
    except:
        # Define empty dictionary for pit sizes of topologies and scenario
        pit_sizes = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        satisfaction_rates = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        print('Extracting IFA effects...')
        # Iterate over each topology received in input
        for topology in topologies:
            # Iterate over each scenario passed as input
            for scenario in scenarios:
                assert scenario in ['normal', 'existing', 'non_existing']
                # Define the path to files containing data for current scenario
                path = simulations_path(download_folder=download_folder,
                                        scenario=scenario,
                                        topology=topology)
                # print('path: {}'.format(path))
                # Get files list containing files of current scenario
                files_list = get_files_list(directory=path, scenario=scenario)
                # If the scenario is not the legitimate one then we need to plot one distribution for each frequency
                # print('scenario: {}'.format(scenario))
                if scenario != 'normal':
                    # Iterate over frequencies
                    frequencies = np.unique([file.split('/')[-3].split('x')[0] for file in files_list])
                    for frequence in frequencies:
                        freq_files = [file for file in files_list if file.split('/')[-3].split('x')[0] == frequence]
                        # Iterate over number of attackers
                        n_atts = set([file.split('/')[-2].split('_')[0] for file in freq_files])
                        for n_att in n_atts:
                            n_att_files = [file for file in freq_files if file.split('/')[-2].split('_')[0] == n_att]
                            # print('n_att_files: {}'.format(n_att_files))
                            # Get pit distributions
                            pits = extract_pits_from_simulation_files(simulation_files=n_att_files,
                                                                      simulation_time=300)
                            # print('pits: {}'.format(pits))
                            # Append distributions to dictionary for plotting
                            pit_sizes[topology][scenario][frequence][n_att] = pits
                            # Get pit distributions
                            srs = extract_srs_from_simulation_files(simulation_files=n_att_files,
                                                                    simulation_time=300)
                            # Append distributions to dictionary for plotting
                            satisfaction_rates[topology][scenario][frequence][n_att] = srs
                else:
                    # print('files_list: {}'.format(files_list))
                    # Get pit distributions
                    pits = extract_pits_from_simulation_files(simulation_files=files_list,
                                                              simulation_time=300)
                    # Append distributions to dictionary for plotting
                    pit_sizes[topology][scenario]['1']['0'] = pits
                    # Get pit distributions
                    srs = extract_srs_from_simulation_files(simulation_files=files_list,
                                                            simulation_time=300)
                    # Append distributions to dictionary for plotting
                    satisfaction_rates[topology][scenario]['1']['0'] = srs
                    # print('pits: {}'.format(pits))
                    # print('srs: {}'.format(srs))
                    # raise RuntimeError()
                # print('pit_sizes: {}'.format(pit_sizes))
                # print('satisfaction_rates: {}'.format(satisfaction_rates))
        # Store pickle files containing pit sizes and satisfaction rates
        file_name = os.path.join(out_path, 'avg_std_pit_sizes_for_time.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(pit_sizes, handle, protocol=pkl.HIGHEST_PROTOCOL)
        file_name = os.path.join(out_path, 'avg_std_satisfaction_rates_for_time.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(satisfaction_rates, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Plot distribution
    print('Plotting...')
    plot_pit_sizes(pit_sizes)
    plot_satisfaction_rate(satisfaction_rates)


def plot_pit_sizes(pit_sizes):
    print('Plotting PIT sizes over time...')
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'ifa_effects')
    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    # print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                    #                                                                     scenario_name,
                    #                                                                     freq_name,
                    #                                                                     n_att_name))
                    # print('data: {}'.format(data))
                    avgs = data['avg']
                    stds = data['std']
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(list(avgs.keys())))
                    router_index = 0
                    for router_name, _ in avgs.items():
                        # if router_name not in ['Rout1', 'Rout5', 'Rout9']:
                        #     continue
                        # print('router_name: {}'.format(router_name))
                        # print('pit_size: {}'.format(pit_size))
                        x = np.linspace(0, len(avgs[router_name]), len(avgs[router_name]))
                        axs.plot(x,
                                 avgs[router_name],
                                 color=colormap(router_index),
                                 linewidth=3,
                                 label=router_name)
                        axs.fill_between(x,
                                         [avgs[router_name][i] - stds[router_name][i] for i in
                                          range(len(avgs[router_name]))],
                                         [avgs[router_name][i] + stds[router_name][i] for i in
                                          range(len(avgs[router_name]))],
                                         color=colormap(router_index),
                                         alpha=0.2)
                        router_index += 1
                    axs.set_ylim(0, 1)
                    axs.set_ylabel('PIT Size')
                    axs.set_xlim(0, 300)
                    axs.set_xlabel('Time (s)')
                    axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                    # Store plot
                    image_name = 'PITS.pdf'
                    image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, image_name), dpi=200)
                    # plt.show()
                    plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    # print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                    #                                                                     scenario_name,
                    #                                                                     freq_name,
                    #                                                                     n_att_name))
                    # print('data: {}'.format(data))
                    avgs = data['avg']
                    stds = data['std']
                    router_index = 0
                    all_routers_names = list(avgs.keys())
                    selected_routers = random.sample(all_routers_names, k=3)
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(selected_routers))
                    for router_name, _ in avgs.items():
                        if router_name not in selected_routers:
                            continue
                        # print('router_name: {}'.format(router_name))
                        # print('pit_size: {}'.format(pit_size))
                        x = np.linspace(0, len(avgs[router_name]), len(avgs[router_name]))
                        axs.plot(x,
                                 avgs[router_name],
                                 color=colormap(router_index),
                                 linewidth=3,
                                 label=router_name)
                        axs.fill_between(x,
                                         [avgs[router_name][i] - stds[router_name][i] for i in
                                          range(len(avgs[router_name]))],
                                         [avgs[router_name][i] + stds[router_name][i] for i in
                                          range(len(avgs[router_name]))],
                                         color=colormap(router_index),
                                         alpha=0.2)
                        router_index += 1
                    axs.set_ylim(0, 1)
                    axs.set_ylabel('PIT Size')
                    axs.set_xlim(0, 300)
                    axs.set_xlabel('Time (s)')
                    axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                    # Store plot
                    image_name = 'PITS of few random routers.pdf'
                    image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, image_name), dpi=200)
                    # plt.show()
                    plt.close()


def plot_satisfaction_rate(pit_sizes):
    print('Plotting satisfaction rates over time...')
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'ifa_effects')
    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    # print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                    #                                                                     scenario_name,
                    #                                                                     freq_name,
                    #                                                                     n_att_name))
                    # print('data: {}'.format(data))
                    avgs = data['avg']
                    # print('avgs: {}'.format(avgs))
                    stds = data['std']
                    # print('stds: {}'.format(stds))
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(list(avgs.keys())))
                    consumer_index = 0
                    for consumer_name, _ in avgs.items():
                        # if consumer_name not in ['Cons2', 'Cons4', 'Cons5']:
                        #     continue
                        # print('consumer_name: {}'.format(consumer_name))
                        # print('satisfaction_rates: {}'.format(satisfaction_rates))
                        # print('avgs[consumer_name]: {}'.format(avgs[consumer_name]))
                        x = np.linspace(0, len(avgs[consumer_name]), len(avgs[consumer_name]))
                        axs.plot(x,
                                 [sr * 100 for sr in avgs[consumer_name]],
                                 color=colormap(consumer_index),
                                 linewidth=3,
                                 label='Cons{}'.format(consumer_index + 1))
                        axs.fill_between(x,
                                         [avgs[consumer_name][i] * 100 - stds[consumer_name][i] * 100 for i in
                                          range(len(avgs[consumer_name]))],
                                         [avgs[consumer_name][i] * 100 + stds[consumer_name][i] * 100 for i in
                                          range(len(avgs[consumer_name]))],
                                         color=colormap(consumer_index),
                                         alpha=0.2)
                        consumer_index += 1
                    axs.set_ylim(0, 100)
                    axs.set_ylabel('Satisfaction Rate (%)')
                    axs.set_xlim(0, 300)
                    axs.set_xlabel('Time (s)')
                    axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                    # Store plot
                    image_name = 'Satisfaction rates.pdf'
                    image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, image_name), dpi=200)
                    # plt.show()
                    plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    # print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                    #                                                                     scenario_name,
                    #                                                                     freq_name,
                    #                                                                     n_att_name))
                    # print('data: {}'.format(data))
                    avgs = data['avg']
                    # print('avgs: {}'.format(avgs))
                    stds = data['std']
                    # print('stds: {}'.format(stds))
                    consumer_index = 0
                    all_consumers_names = list(avgs.keys())
                    selected_consumers = random.sample(all_consumers_names, k=3)
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(selected_consumers))
                    for consumer_name, _ in avgs.items():
                        if consumer_name not in selected_consumers:
                            continue
                        # if consumer_name not in ['Cons2', 'Cons4', 'Cons5']:
                        #     continue
                        # print('consumer_name: {}'.format(consumer_name))
                        # print('satisfaction_rates: {}'.format(satisfaction_rates))
                        # print('avgs[consumer_name]: {}'.format(avgs[consumer_name]))
                        x = np.linspace(0, len(avgs[consumer_name]), len(avgs[consumer_name]))
                        axs.plot(x,
                                 [sr * 100 for sr in avgs[consumer_name]],
                                 color=colormap(consumer_index),
                                 linewidth=3,
                                 label='Cons{}'.format(consumer_index + 1))
                        axs.fill_between(x,
                                         [avgs[consumer_name][i] * 100 - stds[consumer_name][i] * 100 for i in
                                          range(len(avgs[consumer_name]))],
                                         [avgs[consumer_name][i] * 100 + stds[consumer_name][i] * 100 for i in
                                          range(len(avgs[consumer_name]))],
                                         color=colormap(consumer_index),
                                         alpha=0.2)
                        consumer_index += 1
                    axs.set_ylim(0, 100)
                    axs.set_ylabel('Satisfaction Rate (%)')
                    axs.set_xlim(0, 300)
                    axs.set_xlabel('Time (s)')
                    axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                    # Store plot
                    image_name = 'Satisfaction rates of few random consumers.pdf'
                    image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, image_name), dpi=200)
                    # plt.show()
                    plt.close()


def get_cmap(n, name='tab20'):
    return plt.cm.get_cmap(name, n)


def get_empty_dict_from_file_names(download_folder, scenarios, topologies):
    file_list = []
    for scenario in scenarios:
        if scenario == 'normal':
            file_list += glob.glob(
                os.path.join(os.getcwd(), '..', download_folder, scenario, '*', '*', '*', '*', '*.txt'))
        else:
            file_list += glob.glob(
                os.path.join(os.getcwd(), '..', download_folder, 'IFA_4_{}'.format(scenario), '*', '*', '*', '*',
                             '*.txt'))
    # print('file_list: {}'.format(file_list))
    empty_dict = {}
    # topologies = []
    for file in file_list:
        scenario = file.split('/')[-6].split('_')[-1].lower()
        if scenario not in scenarios:
            continue
        topology = file.split('/')[-5].split('_')[0].lower()
        if topology not in topologies:
            continue
        # topologies.append(topology)
        freq = file.split('/')[-3].split('x')[0]
        n_att = file.split('/')[-2].split('_')[0]
        # print('topology: {}, scenario: {}, freq: {}, n_att: {}'.format(topology, scenario, freq, n_att))
        try:
            empty_dict[topology]
        except KeyError:
            empty_dict[topology] = {}
        try:
            empty_dict[topology][scenario]
        except KeyError:
            empty_dict[topology][scenario] = {}
        try:
            empty_dict[topology][scenario][freq]
        except KeyError:
            empty_dict[topology][scenario][freq] = {}
        try:
            empty_dict[topology][scenario][freq][n_att]
        except KeyError:
            empty_dict[topology][scenario][freq][n_att] = []
        # empty_dict.update({topology: {scenario: {freq: {n_att: []}}}})
    #     combinations.append((freq, n_att))
    # print(f'combinations: {combinations}')
    # combinations = list(set(combinations))
    # combinations.append(('1', '0'))
    # print(f'combinations: {combinations}')
    # topologies = list(set(topologies))
    for topology in topologies:
        try:
            empty_dict[topology]['normal']
        except KeyError:
            empty_dict[topology]['normal'] = {}
        try:
            empty_dict[topology]['normal']['1']
        except KeyError:
            empty_dict[topology]['normal']['1'] = {}
        try:
            empty_dict[topology]['normal']['1']['0']
        except KeyError:
            empty_dict[topology]['normal']['1']['0'] = []
    # print('empty_dict: {}'.format(empty_dict))
    return empty_dict


def simulations_path(download_folder, scenario, topology):
    return os.path.join(os.getcwd(), '..', download_folder,
                        'IFA_4_{}'.format(scenario) if scenario != 'normal' else scenario,
                        '{}_topology'.format(topology) if topology != 'dfn' else '{}_topology'.format(topology.upper()))


def get_files_list(directory, scenario):
    # Import stored dictionary of data
    if scenario != 'normal':
        # print('ouaosdngoin: {}'.format(os.path.join(directory, '*', '*', '*', '*.txt')))
        file_names = glob.glob(os.path.join(directory, '*', '*', '*', '*.txt'))
    else:
        file_names = glob.glob(os.path.join(directory, '*.txt'))
    # print('file_names: {}'.format(file_names))
    return file_names


def get_simulation_time(simulation_files, simulation_time=300):
    # print('simulation_files: {}'.format(simulation_files))
    # Check if simulation has run up until the end or not. To avoid NaN issues inside features
    rate_trace_file = [file for file in simulation_files if 'rate-trace' in file][0]
    last_line_of_rate_trace_file = pd.read_csv(rate_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_rate_trace_file = last_line_of_rate_trace_file['Time']
    # Set simulation time depending on the last line of the trace file
    if simulation_time_from_rate_trace_file < simulation_time - 1:
        simulation_time = simulation_time_from_rate_trace_file - 1
    else:
        simulation_time = simulation_time - 1
    # Double check simulation time from the pit trace file
    try:
        pit_trace_file = [file for file in simulation_files if 'format-pit-size' in file][0]
    except IndexError:
        _ = get_data(simulation_files)
        simulation_files = glob.glob(os.path.join('/', os.path.join(*simulation_files[0].split('/')[:-1]), '*.txt'))
        pit_trace_file = [file for file in simulation_files if 'format-pit-size' in file][0]
    last_line_of_pit_trace_file = pd.read_csv(pit_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_pit_trace_file = last_line_of_pit_trace_file['Time'] - 1
    if simulation_time_from_pit_trace_file < simulation_time:
        simulation_time = simulation_time_from_pit_trace_file
    # Double check simulation time from the drop trace file
    drop_trace_file = [file for file in simulation_files if 'drop-trace' in file][0]
    last_line_of_drop_trace_file = pd.read_csv(drop_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_drop_trace_file = last_line_of_drop_trace_file['Time'] - 1
    # print(f'PIT last time: {simulation_time_from_pit_trace_file}')
    # print(f'Rate trace last time: {simulation_time_from_rate_trace_file}')
    # print(f'Drop trace last time: {simulation_time_from_drop_trace_file}')
    if simulation_time_from_drop_trace_file < simulation_time:
        simulation_time = simulation_time_from_drop_trace_file
    # print(f'simulation_time: {simulation_time}')
    return simulation_time


def extract_pits_from_simulation_files(simulation_files, simulation_time=300, avg_std=True):
    # print('simulation_files: {}'.format(simulation_files))
    simulation_files = rename_topology_files(simulation_files)
    # print('simulation_files: {}'.format(simulation_files))
    indices = [int(file.split('.')[-2].split('-')[-1]) for file in simulation_files]
    # print('indices: {}'.format(indices))
    max_sim_index = max(indices)
    # print('max_sim_index: {}'.format(max_sim_index))
    all_pits = {}
    for index in range(1, max_sim_index + 1):
        sim_files = [file for file in simulation_files if int(file.split('.')[-2].split('-')[-1]) == index]
        # print('sim_files: {}'.format(sim_files))
        # Extract data from the considered simulation
        data = get_data(sim_files)
        # Get names of nodes inside a simulation
        routers_names = get_router_names(data)
        # Get simulation time. To avoid NaN issues inside features
        simulation_time = get_simulation_time(sim_files, simulation_time=simulation_time)
        # Define empty list containing all pits found in a simulation
        pits = {}
        #  Iterate over all routers
        for router_name in routers_names:
            pits[router_name] = get_pit_sizes(data, router_name, simulation_time)
        if index == 1:
            for router_name in routers_names:
                all_pits[router_name] = [pits[router_name]]
        else:
            try:
                for router_name in routers_names:
                    all_pits[router_name].append(pits[router_name])
            except KeyError:
                for router_name in routers_names:
                    all_pits[router_name] = [pits[router_name]]
    # print('all_pits: {}'.format(all_pits))
    if avg_std:
        avg_pits = {}
        std_pits = {}
        for router_name in routers_names:
            avg_pits[router_name] = average_list_of_lists_over_time(all_pits[router_name])
            std_pits[router_name] = std_list_of_lists_over_time(all_pits[router_name])
        # print('avg_pits: {}'.format(avg_pits))
        # print('std_pits: {}'.format(std_pits))
        return {'avg': avg_pits, 'std': std_pits}
    else:
        return all_pits


def average_list_of_lists_over_time(list_of_lists):
    min_length = np.min([len(li) for li in list_of_lists])
    array = np.asarray([li[:min_length] for li in list_of_lists])
    # print('array.shape: {}'.format(array.shape))
    avg = np.mean(array, axis=0)
    # print('avg.shape: {}'.format(avg.shape))
    # print('avg.tolist(): {}'.format(avg.tolist()))
    return avg.tolist()


def std_list_of_lists_over_time(list_of_lists):
    min_length = np.min([len(li) for li in list_of_lists])
    # print([li[:min_length] for li in list_of_lists])
    array = np.asarray([li[:min_length] for li in list_of_lists])
    # print('array: {}'.format(array))
    # print('array.shape: {}'.format(array.shape))
    std = np.std(array, axis=0)
    # print('std.shape: {}'.format(std.shape))
    # print('std.tolist(): {}'.format(std.tolist()))
    return std.tolist()


def extract_srs_from_simulation_files(simulation_files, simulation_time=300):
    # print('simulation_files: {}'.format(simulation_files))
    simulation_files = rename_topology_files(simulation_files)
    # print('simulation_files: {}'.format(simulation_files))
    indices = [int(file.split('.')[-2].split('-')[-1]) for file in simulation_files]
    # print('indices: {}'.format(indices))
    max_sim_index = max(indices)
    # print('max_sim_index: {}'.format(max_sim_index))
    all_srss = {}
    for index in range(1, max_sim_index + 1):
        sim_files = [file for file in simulation_files if int(file.split('.')[-2].split('-')[-1]) == index]
        # print('sim_files: {}'.format(sim_files))
        # Extract data from the considered simulation
        data = get_data(sim_files)
        # Get names of nodes inside a simulation
        consumers_names = get_consumers_names(data)
        # Get simulation time. To avoid NaN issues inside features
        simulation_time = get_simulation_time(sim_files, simulation_time=simulation_time)
        # Define empty list containing all pits found in a simulation
        srs = {}
        #  Iterate over all routers
        for consumer_name in consumers_names:
            srs[consumer_name] = get_satisfaction_rates(data, consumer_name, simulation_time)
        if index == 1:
            for consumer_name in consumers_names:
                all_srss[consumer_name] = [srs[consumer_name]]
        else:
            for consumer_name in consumers_names:
                try:
                    all_srss[consumer_name].append(srs[consumer_name])
                except:
                    all_srss[consumer_name] = [srs[consumer_name]]
    # print('all_srss: {}'.format(all_srss))
    avg_srss = {}
    std_srss = {}
    for consumer_name in consumers_names:
        avg_srss[consumer_name] = average_list_of_lists_over_time(all_srss[consumer_name])
        std_srss[consumer_name] = std_list_of_lists_over_time(all_srss[consumer_name])
    # print('avg_srss: {}'.format(avg_srss))
    # print('std_srss: {}'.format(std_srss))
    return {'avg': avg_srss, 'std': std_srss}


def get_consumers_names(data):
    data = remove_topo_data_from_dict(data)
    # Get names of transmitter devices
    consumers_names = data['rate']['Node'].unique()
    # Consider routers only
    consumers_names = [i for i in consumers_names if ('Cons' in i or 'Atta' in i)]
    # print('consumers_names: {}'.format(consumers_names))
    return consumers_names


def get_satisfaction_rates(data, node_name, simulation_time):
    srs = []
    # Define start time as one
    start_time = 1
    # For each index get the corresponding network traffic window and extract the features in that window
    for time in range(start_time, simulation_time + 1):
        # Filer data to get current time window
        filtered_data = filter_data_by_time(data, time)
        satisfaction_rate = get_satisfaction_rate(node_name=node_name,
                                                  data=filtered_data)
        # Add pit sizes to pits
        srs.append(satisfaction_rate)
    return srs


def get_satisfaction_rate(data, node_name):
    # Get different modes of data
    rate_data = data['rate']
    # Get InSatisfiedInterests of consumer at hand
    in_satisfied_interests = \
        rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InSatisfiedInterests')]['PacketRaw']
    in_satisfied_interests = sum(i for i in in_satisfied_interests)
    # Get OutInterests of router at hand
    out_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutInterests')]['PacketRaw']
    out_interests = sum(i for i in out_interests)
    if out_interests == 0 and in_satisfied_interests == 0:
        return 1
    else:
        return min(in_satisfied_interests / max(out_interests, 1), 1)


def get_pit_sizes(data, node_name, simulation_time):
    pits = []
    # Define start time as one
    start_time = 1
    # For each index get the corresponding network traffic window and extract the features in that window
    for time in range(start_time, simulation_time + 1):
        # Filer data to get current time window
        filtered_data = filter_data_by_time(data, time)
        pit_sizes = get_pit_size(node_name=node_name,
                                 data=filtered_data)
        # Add pit sizes to pits
        pits.append(pit_sizes)
    return pits


def get_router_names(data):
    data = remove_topo_data_from_dict(data)
    # Get names of transmitter devices
    routers_names = data['rate']['Node'].unique()
    # Consider routers only
    routers_names = [i for i in routers_names if 'Rout' in i]
    # print('routers_names: {}'.format(routers_names))
    return routers_names


def filter_data_by_time(data, time, verbose=False):
    if verbose:
        print('Time: {}'.format(time))
    data = remove_topo_data_from_dict(data)
    filtered_data = {}
    for key, value in data.items():
        filtered_data[key] = data[key][data[key]['Time'] == time]
    return filtered_data


def get_pit_size(data, node_name):
    # Get different modes of data
    pit_data = data['pit']
    # Get pit size of router at hand
    router_index = node_name.split('Rout')[-1]
    # print('router_index: {}'.format(router_index))
    # print('pit_data: {}'.format(pit_data))
    # print(pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'])
    try:
        pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
    except ValueError:
        pit_size = pit_data[pit_data['Node'] == 'PIT_Rout{}'.format(router_index)]['Size'].item()

    return min(pit_size, 1200) / 1200.


def remove_topo_data_from_dict(data):
    # print(f'data: {data}')
    data = {key: value for key, value in data.items() if key.split('/')[-1].split('-')[0] != 'topology'}
    # print(f'data: {data}')
    return data


def rename_topology_files(files):
    for index, file in enumerate(files):
        if '_topology' in file.split('/')[-1]:
            new_name = os.path.join('/',
                                    os.path.join(*file.split('/')[:-1]),
                                    '{}-{}.txt'.format(file.split('/')[-1].split('.')[0].split('_')[1],
                                                       file.split('/')[-1].split('.')[0].split('_')[0]))
            os.rename(file, new_name)
            files[index] = new_name
    return files


def get_data(files):
    # Define empty dictionary containing data
    data = {}
    # Iterate over all simulation files and read them
    for file in files:
        # print('file: {}'.format(file))
        # Check file type
        file_type = file.split('/')[-1].split('-')[0]
        if file_type == 'format':
            continue
        if file_type == 'pit':
            file = convert_pit_to_decent_format(file)
        if file_type == 'topology':
            # print('Converting topology file to decent format...')
            file = convert_topology_to_decent_format(file)
        # Read csv file
        file_data = pd.read_csv(file, sep='\t', index_col=False)
        # Put data in dictionary
        data[file_type] = file_data
    return data


def get_lines_from_unformatted_topology_file(file):
    reversed_lines = reversed(list(open(file)))
    keep = []
    # Cycle through file from bottom to top
    for line in reversed_lines:
        if line.rstrip() == '':
            continue
        if line.rstrip()[0] == '#':
            break
        else:
            keep.append('\t'.join(line.split()) + '\n')
    keep = reversed(keep)
    return keep


def convert_topology_to_decent_format(file):
    final_lines = get_lines_from_unformatted_topology_file(file)
    # Store file with new name
    new_file = os.path.join('/', os.path.join(*file.split('/')[:-1]),
                            'format-{}'.format(file.split('/')[-1]))
    if os.path.exists(new_file):
        os.remove(new_file)
    with open(new_file, 'w') as f:
        f.write('Source\tDestination\n')
        for item in final_lines:
            f.write(item)
    return new_file


def convert_pit_to_decent_format(file):
    # Get number of lines of file
    num_lines = sum(1 for _ in open(file))
    # Get index of lines containing time
    times_lines = [i for i, line in enumerate(open(file)) if 'Simulation time' in line]
    # Get number of routers
    n_routers = times_lines[1] - 1
    # Get lines containing actual data
    data_lines = [line for line in open(file) if 'Simulation time' not in line]
    # Append timing to each line of the data_lines
    final_lines = []
    for i, line in enumerate(open(file)):
        if 'Simulation time' not in line:
            time = i // (n_routers + 1) + 1
            final_lines.append(str(time) + '\t' + line.replace(' ', '\t'))
    # Store file with new name
    new_file = os.path.join(*file.split('/')[:-1])
    new_file = os.path.join('/', new_file, 'format-{}'.format(file.split('/')[-1]))
    if os.path.exists(new_file):
        os.remove(new_file)
    with open(new_file, 'w') as f:
        f.write('Time\tNode\tWord1\tWord2\tWord3\tSize\n')
        for item in final_lines:
            f.write(item)
    return new_file


def get_graph_structure(topology_lines_dataframe, debug=False):
    # Open file containing train_topology structure
    # topology_file = os.path.join(self.data_dir, 'topologies', '{}_topology.txt'.format(self.topology))
    # router_links = []
    # for line in open(topology_file, 'r'):
    #     source = line.split(',')[0]
    #     dest = line.split(',')[1].split('\n')[0]
    #     if source[:4] == 'Rout' and dest[:4] == 'Rout':
    #         router_links.append([source[4:], dest[4:]])
    topology_lines_dataframe = topology_lines_dataframe.reset_index()  # make sure indexes pair with number of rows
    router_links = []
    all_nodes_names = []
    for index, row in topology_lines_dataframe.iterrows():
        source = row['Source']
        dest = row['Destination']
        router_links.append([source, dest])
        # router_links.append([dest, source])
        if source not in all_nodes_names:
            all_nodes_names.append(source)
        if dest not in all_nodes_names:
            all_nodes_names.append(dest)
    # print('all_nodes_names: {}'.format(all_nodes_names))
    # Use nx to obtain the graph corresponding to the graph
    graph = nx.Graph()
    # Build the DODAG graph from nodes and edges lists
    graph.add_nodes_from(all_nodes_names)
    graph.add_edges_from(router_links)
    # Print and plot for debugging purposes
    if debug:
        print('graph: {}'.format(graph))
        print('graph.nodes: {}'.format(graph.nodes))
        print('graph.edges: {}'.format(graph.edges))
        subax1 = plt.subplot(111)
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()
    # Return networkx graph
    return graph


def plot_ifa_vs_distance(download_folder, scenarios, topologies):
    # Check if pickle files are available for the pit sizes and satisfaction rates
    out_path = os.path.join(os.getcwd(), '..', 'output', 'pickles', 'ifa_effects')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        file_name = os.path.join(out_path, 'pit_sizes_for_distance.pkl')
        with open(file_name, 'rb') as handle:
            pit_sizes = pkl.load(handle)
    except FileNotFoundError:
        # Define empty dictionary for pit sizes of topologies and scenario
        pit_sizes = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        satisfaction_rates = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        print('Extracting IFA effects...')
        # Iterate over each topology received in input
        for topology in topologies:
            # Iterate over each scenario passed as input
            for scenario in scenarios:
                assert scenario in ['normal', 'existing', 'non_existing']
                # Define the path to files containing data for current scenario
                path = simulations_path(download_folder=download_folder,
                                        scenario=scenario,
                                        topology=topology)
                # print('path: {}'.format(path))
                # Get files list containing files of current scenario
                files_list = get_files_list(directory=path, scenario=scenario)
                # If the scenario is not the legitimate one then we need to plot one distribution for each frequency
                # print('scenario: {}'.format(scenario))
                if scenario != 'normal':
                    # Iterate over frequencies
                    frequencies = np.unique([file.split('/')[-3].split('x')[0] for file in files_list])
                    for frequence in frequencies:
                        freq_files = [file for file in files_list if file.split('/')[-3].split('x')[0] == frequence]
                        # Iterate over number of attackers
                        n_atts = set([file.split('/')[-2].split('_')[0] for file in freq_files])
                        for n_att in n_atts:
                            n_att_files = [file for file in freq_files if file.split('/')[-2].split('_')[0] == n_att]

                            print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topology,
                                                                                                scenario,
                                                                                                frequence,
                                                                                                n_att))
                            # print('n_att_files: {}'.format(n_att_files))
                            # Get pit distributions
                            pits = extract_pits_vs_distance_from_simulation_files(simulation_files=n_att_files,
                                                                                  simulation_time=300)
                            # print('pits: {}'.format(pits))
                            # Append distributions to dictionary for plotting
                            pit_sizes[topology][scenario][frequence][n_att] = pits
                else:
                    raise RuntimeError('Scenario should not be normal when considering pit vs distance from attacker!')
                # print('pit_sizes: {}'.format(pit_sizes))
                # print('satisfaction_rates: {}'.format(satisfaction_rates))
        # Store pickle files containing pit sizes and satisfaction rates
        file_name = os.path.join(out_path, 'pit_sizes_for_distance.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(pit_sizes, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Plot distribution
    print('Plotting...')
    plot_pit_vs_distance(pit_sizes)


def extract_pits_vs_distance_from_simulation_files(simulation_files, simulation_time=300):
    # print('simulation_files: {}'.format(simulation_files))
    simulation_files = rename_topology_files(simulation_files)
    # print('simulation_files: {}'.format(simulation_files))
    indices = [int(file.split('.')[-2].split('-')[-1]) for file in simulation_files]
    # print('indices: {}'.format(indices))
    max_sim_index = max(indices)
    # print('max_sim_index: {}'.format(max_sim_index))
    all_pits = {}
    for index in range(1, max_sim_index + 1):
        sim_files = [file for file in simulation_files if int(file.split('.')[-2].split('-')[-1]) == index]
        # print('sim_files: {}'.format(sim_files))
        # Extract data from the considered simulation
        data = get_data(sim_files)
        # Get names of nodes inside a simulation
        routers_names = get_router_names(data)
        # Get names of nodes inside a simulation
        attackers_names = get_attackers_names(data)
        # print('attackers_names: {}'.format(attackers_names))
        # Get simulation time. To avoid NaN issues inside features
        simulation_time = get_simulation_time(sim_files, simulation_time=simulation_time)
        # Get dictionary containing distances between all nodes in the topology
        distances_dict = get_distances_dict(data)
        # print('distances_dict: {}'.format(distances_dict))
        # Define empty list containing all pits found in a simulation
        pits = {}
        #  Iterate over all routers
        for router_name in routers_names:
            pits[router_name] = get_pit_sizes_vs_distance(data, router_name, simulation_time,
                                                          distances_dict, attackers_names)
        if index == 1:
            for router_name in routers_names:
                all_pits[router_name] = [pits[router_name]]
        else:
            try:
                for router_name in routers_names:
                    all_pits[router_name].append(pits[router_name])
            except KeyError:
                for router_name in routers_names:
                    all_pits[router_name] = [pits[router_name]]
    return all_pits


def get_attackers_names(data):
    data = remove_topo_data_from_dict(data)
    # Get names of transmitter devices
    attackers_names = data['rate']['Node'].unique()
    # Consider routers only
    attackers_names = [i for i in attackers_names if 'Atta' in i]
    # print('attackers_names: {}'.format(attackers_names))
    return attackers_names


def get_pit_sizes_vs_distance(data, node_name, simulation_time, distances_dict, attackers_names):
    pits = []
    # Define start time as one
    start_time = 51
    # For each index get the corresponding network traffic window and extract the features in that window
    for time in range(start_time, simulation_time + 1):
        # Filer data to get current time window
        filtered_data = filter_data_by_time(data, time)
        pit_sizes = get_pit_size(node_name=node_name,
                                 data=filtered_data)
        # Add pit sizes to pits
        pits.append(pit_sizes)
    # print('distances_dict: {}'.format(distances_dict))
    min_distance_from_attacker = 10000
    for attacker_name in attackers_names:
        # print('node_name: {}, attacker_name: {}'.format(node_name, attacker_name))
        try:
            dist = distances_dict[node_name][attacker_name]
            if dist < min_distance_from_attacker:
                min_distance_from_attacker = dist
        except KeyError:
            continue
    print('min_distance between {} and any attacker is: {}'.format(node_name, min_distance_from_attacker))
    return {'pits': pits, 'dist': min_distance_from_attacker}


def get_distances_dict(data):
    topology = get_graph_structure(data['topology'])
    return dict(nx.all_pairs_shortest_path_length(topology))


def plot_pit_vs_distance(pit_sizes):
    print('Plotting PIT sizes against attacker distance...')
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'ifa_effects')
    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        # print('topo_name: {}'.format(topo_name))
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            if scenario_name == 'normal':
                continue
            # print('scenario_name: {}'.format(scenario_name))
            n_points = 0
            all_distances = []
            for freq_name, freq_dict in scenario_dict.items():
                for n_att_name, data in freq_dict.items():
                    n_points += 1
                    # print('data: {}'.format(data))
                    for router_name, router_data in data.items():
                        for single_run_data in router_data:
                            all_distances.append(single_run_data['dist'])
            all_distances = list(set(all_distances))
            # print('all_distances: {}'.format(all_distances))
            # print('n_points: {}'.format(n_points))
            # fig, axs = plt.subplots(figsize=(15, 20))
            # colormap = get_cmap(n_points)
            index = 0
            for freq_name, freq_dict in scenario_dict.items():
                for n_att_name, data in freq_dict.items():
                    index += 1
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(all_distances), 'Accent')
                    pit_distances_dict = {dist: [] for dist in all_distances}
                    for router_name, router_data in data.items():
                        for single_run_data in router_data:
                            pit_distances_dict[single_run_data['dist']] += single_run_data['pits']
                    # print('pit_distances_dict: {}'.format(pit_distances_dict))
                    avg_pit_distances_dict = [np.mean(pit_distances_dict[i]) for i in all_distances]
                    std_pit_distances_dict = [np.std(pit_distances_dict[i]) for i in all_distances]
                    # print('avg_pit_distances_dict: {}'.format(avg_pit_distances_dict))
                    # print('std_pit_distances_dict: {}'.format(std_pit_distances_dict))
                    axs.plot(all_distances,
                             avg_pit_distances_dict,
                             color=colormap(index),
                             linewidth=3,
                             label='F={}x and N={}'.format(freq_name, n_att_name))
                    axs.fill_between(all_distances,
                                     [avg_pit_distances_dict[i] - std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     [avg_pit_distances_dict[i] + std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     alpha=0.2)
                    axs.set_ylim(0, 1)
                    axs.set_ylabel('PIT Size')
                    axs.set_xlim(1, max(all_distances))
                    axs.set_xlabel('Minimum distance from an attacker')
                    axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                    # Store plot
                    image_name = 'PITS_vs_distance.pdf'
                    image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_path, image_name), dpi=200)
                    # plt.show()
                    plt.close()

            for freq_name, freq_dict in scenario_dict.items():
                index = 0
                fig, axs = plt.subplots(figsize=(15, 20))
                colormap = get_cmap(len(list(freq_dict.keys())), 'Accent')
                for n_att_name, data in freq_dict.items():
                    pit_distances_dict = {dist: [] for dist in all_distances}
                    for router_name, router_data in data.items():
                        for single_run_data in router_data:
                            pit_distances_dict[single_run_data['dist']] += single_run_data['pits']
                    # print('pit_distances_dict: {}'.format(pit_distances_dict))
                    avg_pit_distances_dict = [np.mean(pit_distances_dict[i]) for i in all_distances]
                    std_pit_distances_dict = [np.std(pit_distances_dict[i]) for i in all_distances]
                    # print('avg_pit_distances_dict: {}'.format(avg_pit_distances_dict))
                    # print('std_pit_distances_dict: {}'.format(std_pit_distances_dict))
                    axs.plot(all_distances,
                             avg_pit_distances_dict,
                             color=colormap(index),
                             linewidth=3,
                             label='N={}'.format(n_att_name))
                    axs.fill_between(all_distances,
                                     [avg_pit_distances_dict[i] - std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     [avg_pit_distances_dict[i] + std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     alpha=0.2, color=colormap(index))
                    index += 1
                axs.set_ylim(0, 1)
                axs.set_ylabel('PIT Size')
                axs.set_xlim(1, max(all_distances))
                axs.set_xlabel('Minimum distance from an attacker')
                axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                # Store plot
                image_name = 'PITS_vs_distance.pdf'
                image_path = os.path.join(out_path, topo_name, scenario_name, freq_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                plt.tight_layout()
                plt.savefig(os.path.join(image_path, image_name), dpi=200)
                # plt.show()
                plt.close()

            freqs = [freq_name for freq_name, _ in scenario_dict.items()]
            n_atts = [n_att_name for n_att_name, _ in scenario_dict[freqs[0]].items()]
            # print('freqs: {}'.format(freqs))
            # print('n_atts: {}'.format(n_atts))
            for n_att_name in n_atts:
                index = 0
                fig, axs = plt.subplots(figsize=(15, 20))
                colormap = get_cmap(len(freqs), 'Accent')
                for freq in freqs:
                    data = scenario_dict[freq][n_att_name]
                    pit_distances_dict = {dist: [] for dist in all_distances}
                    for router_name, router_data in data.items():
                        for single_run_data in router_data:
                            pit_distances_dict[single_run_data['dist']] += single_run_data['pits']
                    # print('pit_distances_dict: {}'.format(pit_distances_dict))
                    # for i in all_distances:
                    #     print('distance {} -> number of samples: {}'.format(i, len(pit_distances_dict[i])))
                    avg_pit_distances_dict = [np.mean(pit_distances_dict[i]) for i in all_distances]
                    std_pit_distances_dict = [np.std(pit_distances_dict[i]) for i in all_distances]
                    # print('avg_pit_distances_dict: {}'.format(avg_pit_distances_dict))
                    # print('std_pit_distances_dict: {}'.format(std_pit_distances_dict))
                    axs.plot(all_distances,
                             avg_pit_distances_dict,
                             color=colormap(index),
                             linewidth=3,
                             label='F={}x'.format(freq))
                    axs.fill_between(all_distances,
                                     [avg_pit_distances_dict[i] - std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     [avg_pit_distances_dict[i] + std_pit_distances_dict[i] for i in
                                      range(len(avg_pit_distances_dict))],
                                     alpha=0.2, color=colormap(index))
                    index += 1
                axs.set_ylim(0, 1)
                axs.set_ylabel('PIT Size')
                axs.set_xlim(1, max(all_distances))
                axs.set_xlabel('Minimum distance from an attacker')
                axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                # Store plot
                image_name = 'PITS_vs_distance_n={}.pdf'.format(n_att_name)
                image_path = os.path.join(out_path, topo_name, scenario_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                plt.tight_layout()
                plt.savefig(os.path.join(image_path, image_name), dpi=200)
                # plt.show()
                plt.close()

            fig, axs = plt.subplots(figsize=(15, 20))
            colormap = get_cmap(n_points)
            index = 0
            for freq_name, freq_dict in scenario_dict.items():
                for n_att_name, data in freq_dict.items():
                    index += 1
                    pit_distances_dict = {dist: [] for dist in all_distances}
                    for router_name, router_data in data.items():
                        for single_run_data in router_data:
                            pit_distances_dict[single_run_data['dist']] += single_run_data['pits']
                    # print('pit_distances_dict: {}'.format(pit_distances_dict))
                    avg_pit_distances_dict = [np.mean(pit_distances_dict[i]) for i in all_distances]
                    std_pit_distances_dict = [np.std(pit_distances_dict[i]) for i in all_distances]
                    # print('avg_pit_distances_dict: {}'.format(avg_pit_distances_dict))
                    # print('std_pit_distances_dict: {}'.format(std_pit_distances_dict))
                    axs.plot(all_distances,
                             avg_pit_distances_dict,
                             color=colormap(index),
                             linewidth=3,
                             label='F={}x and N={}'.format(freq_name, n_att_name))
                    # axs.fill_between(all_distances,
                    #                  [avg_pit_distances_dict[i] - std_pit_distances_dict[i] for i in
                    #                   range(len(avg_pit_distances_dict))],
                    #                  [avg_pit_distances_dict[i] + std_pit_distances_dict[i] for i in
                    #                   range(len(avg_pit_distances_dict))],
                    #                  alpha=0.2)
            axs.set_ylim(0, 1)
            axs.set_ylabel('PIT Size')
            axs.set_xlim(0, max(all_distances) + 1)
            axs.set_xlabel('Minimum distance from an attacker')
            axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
            # Store plot
            image_name = 'PITS_vs_distance.pdf'
            image_path = os.path.join(out_path, topo_name, scenario_name)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            plt.tight_layout()
            plt.savefig(os.path.join(image_path, image_name), dpi=200)
            # plt.show()
            plt.close()


def plot_std_vs_time(download_folder, scenarios, topologies, interval=10):
    # Check if pickle files are available for the pit sizes and satisfaction rates
    out_path = os.path.join(os.getcwd(), '..', 'output', 'pickles', 'ifa_effects')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        file_name = os.path.join(out_path, 'pit_sizes_for_std_vs_time.pkl')
        with open(file_name, 'rb') as handle:
            pit_sizes = pkl.load(handle)
    except FileNotFoundError:
        # Define empty dictionary for pit sizes of topologies and scenario
        pit_sizes = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        satisfaction_rates = get_empty_dict_from_file_names(download_folder, scenarios, topologies)
        print('Extracting IFA effects...')
        # Iterate over each topology received in input
        for topology in topologies:
            # Iterate over each scenario passed as input
            for scenario in scenarios:
                assert scenario in ['normal', 'existing', 'non_existing']
                # Define the path to files containing data for current scenario
                path = simulations_path(download_folder=download_folder,
                                        scenario=scenario,
                                        topology=topology)
                # print('path: {}'.format(path))
                # Get files list containing files of current scenario
                files_list = get_files_list(directory=path, scenario=scenario)
                # If the scenario is not the legitimate one then we need to plot one distribution for each frequency
                if scenario != 'normal':
                    # Iterate over frequencies
                    frequencies = np.unique([file.split('/')[-3].split('x')[0] for file in files_list])
                    for frequence in frequencies:
                        freq_files = [file for file in files_list if file.split('/')[-3].split('x')[0] == frequence]
                        # Iterate over number of attackers
                        n_atts = set([file.split('/')[-2].split('_')[0] for file in freq_files])
                        for n_att in n_atts:
                            print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topology,
                                                                                                scenario,
                                                                                                frequence,
                                                                                                n_att))
                            n_att_files = [file for file in freq_files if file.split('/')[-2].split('_')[0] == n_att]
                            # print('n_att_files: {}'.format(n_att_files))
                            # Get pit distributions
                            pits = extract_pits_from_simulation_files(simulation_files=n_att_files,
                                                                      simulation_time=300,
                                                                      avg_std=False)
                            # print('pits: {}'.format(pits))
                            # Append distributions to dictionary for plotting
                            pit_sizes[topology][scenario][frequence][n_att] = pits
                else:
                    raise RuntimeError('Scenario should not be normal when considering pit vs distance from attacker!')
                # print('pit_sizes: {}'.format(pit_sizes))
                # print('satisfaction_rates: {}'.format(satisfaction_rates))
        # Store pickle files containing pit sizes and satisfaction rates
        file_name = os.path.join(out_path, 'pit_sizes_for_std_vs_time.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(pit_sizes, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Plot distribution
    print('Plotting...')
    _plot_std_vs_time(pit_sizes, interval=interval)


def _plot_std_vs_time(pit_sizes, interval=10):
    print('Plotting standard deviation over time...')
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'ifa_effects')
    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                                                                                        scenario_name,
                                                                                        freq_name,
                                                                                        n_att_name))
                    # print('data: {}'.format(data))
                    if data != []:
                        fig, axs = plt.subplots(figsize=(15, 20))
                        colormap = get_cmap(len(list(data.keys())))
                        router_index = 0
                        all_stds = {}
                        for router_name, list_of_runs in data.items():
                            single_run = list_of_runs[0]
                            stds = [np.std(single_run[i-interval: i]) for i in range(interval, len(single_run))]
                            all_stds[router_name] = stds
                            # print('stds: {}'.format(stds))
                            # print('router_name: {}'.format(router_name))
                            # print('pit_size: {}'.format(pit_size))
                            x = np.linspace(interval, len(single_run), len(stds))
                            axs.plot(x,
                                     stds,
                                     color=colormap(router_index),
                                     linewidth=3,
                                     label=router_name)
                            router_index += 1
                        # axs.set_ylim(0, 1)
                        axs.set_ylabel('Standard deviation')
                        axs.set_xlim(0, 300)
                        axs.set_xlabel('Time (s)')
                        axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                        # Store plot
                        image_name = 'STD_vs_TIME.pdf'
                        image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_path, image_name), dpi=200)
                        # plt.show()
                        plt.close()

                        min_length = min([len(li) for li in list(all_stds.values())])
                        all_stds_values = [li[:min_length] for li in list(all_stds.values())]

                        avg_stds = np.mean(np.asarray(all_stds_values), axis=0) # .tolist()
                        # print('all_stds: {}'.format(all_stds))
                        # print('np.asarray(all_stds_values): {}'.format(np.asarray(all_stds_values)))
                        # print('avg_stds: {}'.format(avg_stds))
                        # print('x.shape: {}'.format(x.shape))
                        # print('avg_stds.shape: {}'.format(avg_stds.shape))
                        fig, axs = plt.subplots(figsize=(15, 20))
                        axs.plot(x,
                                 avg_stds,
                                 color=colormap(router_index),
                                 linewidth=3,
                                 label='avg-std over all routers')
                        router_index += 1
                        axs.set_ylim(0, 1)
                        axs.set_ylabel('|Standard deviation|')
                        axs.set_xlim(0, 300)
                        axs.set_xlabel('Time (s)')
                        axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                        # Store plot
                        image_name = 'AVG_STD_vs_TIME.pdf'
                        image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_path, image_name), dpi=200)
                        # plt.show()
                        plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                                                                                        scenario_name,
                                                                                        freq_name,
                                                                                        n_att_name))
                    # print('data: {}'.format(data))
                    if data != []:
                        fig, axs = plt.subplots(figsize=(15, 20))
                        colormap = get_cmap(len(list(data.keys())))
                        router_index = 0
                        all_stds = {}
                        for router_name, list_of_runs in data.items():
                            stds_over_runs = []
                            for single_run in list_of_runs:
                                stds = [np.std(single_run[i - interval: i]) for i in range(interval, len(single_run))]
                                stds_over_runs.append(stds)
                            min_length = np.min([len(li) for li in stds_over_runs])
                            stds_over_runs = [li[:min_length] for li in stds_over_runs]
                            # print('stds_over_runs: {}'.format(stds_over_runs))
                            # print('np.asarray(stds_over_runs): {}'.format(np.asarray(stds_over_runs)))
                            all_stds[router_name] = {'avg': np.mean(np.asarray(stds_over_runs), axis=0).tolist(),
                                                     'std': np.std(np.asarray(stds_over_runs), axis=0).tolist()}
                            # print('all_stds: {}'.format(all_stds))
                            # print('router_name: {}'.format(router_name))
                            # print('pit_size: {}'.format(pit_size))
                            x = np.linspace(interval, len(single_run), len(all_stds[router_name]['avg']))
                            axs.plot(x,
                                     all_stds[router_name]['avg'],
                                     color=colormap(router_index),
                                     linewidth=3,
                                     label=router_name)
                            axs.fill_between(x,
                                             [all_stds[router_name]['avg'][i] - all_stds[router_name]['std'][i] for i in
                                              range(len(all_stds[router_name]['avg']))],
                                             [all_stds[router_name]['avg'][i] + all_stds[router_name]['std'][i] for i in
                                              range(len(all_stds[router_name]['avg']))],
                                             color=colormap(router_index),
                                             alpha=0.2)
                            router_index += 1
                        # axs.set_ylim(0, 1)
                        axs.set_ylabel('PIT Size Rolling S.D.')
                        axs.set_xlim(0, 300)
                        axs.set_xlabel('Time (s)')
                        axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                        # Store plot
                        image_name = 'STD_vs_TIME_averaged_over_runs.pdf'
                        image_path = os.path.join(out_path, topo_name, scenario_name, freq_name, n_att_name)
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)
                        plt.tight_layout()
                        plt.savefig(os.path.join(image_path, image_name), dpi=200)
                        # plt.show()
                        plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            setup_index = 0
            fig, axs = plt.subplots(figsize=(20, 20))
            n_setups = 0
            for freq_name, freq_dict in scenario_dict.items():
                for n_att_name, data in freq_dict.items():
                    n_setups += 1
            colormap = get_cmap(n_setups)
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                                                                                        scenario_name,
                                                                                        freq_name,
                                                                                        n_att_name))
                    # print('data: {}'.format(data))
                    if data != []:
                        router_index = 0
                        all_stds = {}
                        for router_name, list_of_runs in data.items():
                            stds_over_runs = []
                            for single_run in list_of_runs:
                                stds = [np.std(single_run[i - interval: i]) for i in
                                        range(interval, len(single_run))]
                                stds_over_runs.append(stds)
                            min_length = np.min([len(li) for li in stds_over_runs])
                            stds_over_runs = [li[:min_length] for li in stds_over_runs]
                            # print('stds_over_runs: {}'.format(stds_over_runs))
                            # print('np.asarray(stds_over_runs): {}'.format(np.asarray(stds_over_runs)))
                            all_stds[router_name] = np.mean(np.asarray(stds_over_runs), axis=0).tolist()
                            # print('all_stds: {}'.format(all_stds))
                            # print('router_name: {}'.format(router_name))
                            # print('pit_size: {}'.format(pit_size))
                        min_length = min([len(li) for li in list(all_stds.values())])
                        all_stds_values = [li[:min_length] for li in list(all_stds.values())]
                        avg_of_stds = np.mean(np.asarray(all_stds_values), axis=0).tolist()
                        std_of_stds = np.std(np.asarray(all_stds_values), axis=0).tolist()
                        x = np.linspace(interval, len(avg_of_stds)+interval, len(avg_of_stds))
                        axs.plot(x,
                                 avg_of_stds,
                                 color=colormap(setup_index),
                                 linewidth=3,
                                 label='N={} F={}x'.format(n_att_name, freq_name))
                        axs.fill_between(x,
                                         [avg_of_stds[i] - std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         [avg_of_stds[i] + std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         color=colormap(setup_index),
                                         alpha=0.2)
                        setup_index += 1
            # axs.set_ylim(0, 1)
            axs.set_ylabel('PIT Size Rolling S.D.')
            axs.set_xlim(0, 300)
            axs.set_xlabel('Time (s)')
            axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
            # Store plot
            image_name = 'STD_vs_TIME_over_all_setups.pdf'
            image_path = os.path.join(out_path, topo_name, scenario_name)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            plt.tight_layout()
            plt.savefig(os.path.join(image_path, image_name), dpi=200)
            # plt.show()
            plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                setup_index = 0
                fig, axs = plt.subplots(figsize=(15, 20))
                n_setups = 0
                for n_att_name, data in freq_dict.items():
                    n_setups += 1
                colormap = get_cmap(n_setups)
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                                                                                        scenario_name,
                                                                                        freq_name,
                                                                                        n_att_name))
                    # print('data: {}'.format(data))
                    if data != []:
                        router_index = 0
                        all_stds = {}
                        for router_name, list_of_runs in data.items():
                            stds_over_runs = []
                            for single_run in list_of_runs:
                                stds = [np.std(single_run[i - interval: i]) for i in
                                        range(interval, len(single_run))]
                                stds_over_runs.append(stds)
                            min_length = np.min([len(li) for li in stds_over_runs])
                            stds_over_runs = [li[:min_length] for li in stds_over_runs]
                            # print('stds_over_runs: {}'.format(stds_over_runs))
                            # print('np.asarray(stds_over_runs): {}'.format(np.asarray(stds_over_runs)))
                            all_stds[router_name] = np.mean(np.asarray(stds_over_runs), axis=0).tolist()
                            # print('all_stds: {}'.format(all_stds))
                            # print('router_name: {}'.format(router_name))
                            # print('pit_size: {}'.format(pit_size))
                        min_length = min([len(li) for li in list(all_stds.values())])
                        all_stds_values = [li[:min_length] for li in list(all_stds.values())]
                        avg_of_stds = np.mean(np.asarray(all_stds_values), axis=0).tolist()
                        std_of_stds = np.std(np.asarray(all_stds_values), axis=0).tolist()
                        x = np.linspace(interval, len(avg_of_stds) + interval, len(avg_of_stds))
                        axs.plot(x,
                                 avg_of_stds,
                                 color=colormap(setup_index),
                                 linewidth=3,
                                 label='N={}'.format(n_att_name))
                        axs.fill_between(x,
                                         [avg_of_stds[i] - std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         [avg_of_stds[i] + std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         color=colormap(setup_index),
                                         alpha=0.2)
                        setup_index += 1
                # axs.set_ylim(0, 1)
                axs.set_ylabel('PIT Size Rolling S.D.')
                axs.set_xlim(0, 300)
                axs.set_xlabel('Time (s)')
                axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                # Store plot
                image_name = 'STD_vs_TIME_over_all_number_of_attackers.pdf'
                image_path = os.path.join(out_path, topo_name, scenario_name, freq_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                plt.tight_layout()
                plt.savefig(os.path.join(image_path, image_name), dpi=200)
                # plt.show()
                plt.close()

    # Iterate over each topology received in input
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            freq_names = [freq_name for freq_name, _ in scenario_dict.items()]
            n_att_names = [n_att_name for n_att_name, _ in scenario_dict[freq_names[0]].items()]
            for n_att_name in n_att_names:
                setup_index = 0
                fig, axs = plt.subplots(figsize=(15, 20))
                n_setups = 0
                for freq_name in freq_names:
                    n_setups += 1
                colormap = get_cmap(n_setups)
                # Iterate over number of attackers
                for freq_name in freq_names:
                    print('topology: {}, scenario: {}, frequency: {}, n_att: {}'.format(topo_name,
                                                                                        scenario_name,
                                                                                        freq_name,
                                                                                        n_att_name))
                    data = scenario_dict[freq_name][n_att_name]
                    # print('data: {}'.format(data))
                    if data != []:
                        router_index = 0
                        all_stds = {}
                        for router_name, list_of_runs in data.items():
                            stds_over_runs = []
                            for single_run in list_of_runs:
                                stds = [np.std(single_run[i - interval: i]) for i in
                                        range(interval, len(single_run))]
                                stds_over_runs.append(stds)
                            min_length = np.min([len(li) for li in stds_over_runs])
                            stds_over_runs = [li[:min_length] for li in stds_over_runs]
                            # print('stds_over_runs: {}'.format(stds_over_runs))
                            # print('np.asarray(stds_over_runs): {}'.format(np.asarray(stds_over_runs)))
                            all_stds[router_name] = np.mean(np.asarray(stds_over_runs), axis=0).tolist()
                            # print('all_stds: {}'.format(all_stds))
                            # print('router_name: {}'.format(router_name))
                            # print('pit_size: {}'.format(pit_size))
                        min_length = min([len(li) for li in list(all_stds.values())])
                        all_stds_values = [li[:min_length] for li in list(all_stds.values())]
                        avg_of_stds = np.mean(np.asarray(all_stds_values), axis=0).tolist()
                        std_of_stds = np.std(np.asarray(all_stds_values), axis=0).tolist()
                        x = np.linspace(interval, len(avg_of_stds) + interval, len(avg_of_stds))
                        axs.plot(x,
                                 avg_of_stds,
                                 color=colormap(setup_index),
                                 linewidth=3,
                                 label='F={}x'.format(freq_name))
                        axs.fill_between(x,
                                         [avg_of_stds[i] - std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         [avg_of_stds[i] + std_of_stds[i] for
                                          i in
                                          range(len(avg_of_stds))],
                                         color=colormap(setup_index),
                                         alpha=0.2)
                        setup_index += 1
                # axs.set_ylim(0, 1)
                axs.set_ylabel('PIT Size Rolling S.D.')
                axs.set_xlim(0, 300)
                axs.set_xlabel('Time (s)')
                axs.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
                # Store plot
                image_name = 'STD_vs_TIME_over_all_frequencies_of_attack_N={}.pdf'.format(n_att_name)
                image_path = os.path.join(out_path, topo_name, scenario_name)
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                plt.tight_layout()
                plt.savefig(os.path.join(image_path, image_name), dpi=200)
                # plt.show()
                plt.close()


def main():
    # Define scenarios for which the distribution plot is required
    download_folder = 'ifa_data'
    # Define scenarios for which the distribution plot is required
    scenario = ['normal', 'existing']
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn', 'large']
    # Run distribution plotter
    plot_ifa_effects(download_folder, scenario, topologies)
    scenario = ['existing']
    plot_ifa_vs_distance(download_folder, scenario, topologies)
    plot_std_vs_time(download_folder, scenario, topologies, interval=10)


if __name__ == '__main__':
    main()
