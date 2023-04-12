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
import argparse
import itertools
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import warnings

SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIG_SIZE = 40
plt.rc('font', size=BIG_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
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
        file_name = os.path.join(out_path, 'pit_sizes.pkl')
        with open(file_name, 'rb') as handle:
            pit_sizes = pkl.load(handle)
        file_name = os.path.join(out_path, 'satisfaction_rates.pkl')
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
        file_name = os.path.join(out_path, 'pit_sizes.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(pit_sizes, handle, protocol=pkl.HIGHEST_PROTOCOL)
        file_name = os.path.join(out_path, 'satisfaction_rates.pkl')
        with open(file_name, 'wb') as handle:
            pkl.dump(satisfaction_rates, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Plot distribution
    print('Plotting...')
    plot_pit_sizes(pit_sizes)
    plot_satisfaction_rate(satisfaction_rates)


def plot_pit_sizes(pit_sizes):
    print('Plotting PIT sizes...')
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
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(list(data.keys())))
                    router_index = 0
                    for router_name, pit_size in data.items():
                        # print('router_name: {}'.format(router_name))
                        # print('pit_size: {}'.format(pit_size))
                        x = np.linspace(0, len(pit_size), len(pit_size))
                        axs.plot(x, pit_size,
                                 color=colormap(router_index), linewidth=3,
                                 label=router_name)
                        router_index += 1
                    axs.set_ylim(0, 1200)
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


def plot_satisfaction_rate(pit_sizes):
    print('Plotting satisfaction rates...')
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
                    fig, axs = plt.subplots(figsize=(15, 20))
                    colormap = get_cmap(len(list(data.keys())))
                    consumer_index = 0
                    for consumer_name, satisfaction_rates in data.items():
                        # print('consumer_name: {}'.format(consumer_name))
                        # print('satisfaction_rates: {}'.format(satisfaction_rates))
                        x = np.linspace(0, len(satisfaction_rates), len(satisfaction_rates))
                        axs.plot(x, [sr*100 for sr in satisfaction_rates],
                                 color=colormap(consumer_index), linewidth=3,
                                 label='Cons{}'.format(consumer_index+1))
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


def extract_pits_from_simulation_files(simulation_files, simulation_time=300):
    # print('simulation_files: {}'.format(simulation_files))
    simulation_files = rename_topology_files(simulation_files)
    # Extract data from the considered simulation
    data = get_data(simulation_files)
    # Get names of nodes inside a simulation
    routers_names = get_router_names(data)
    # Get simulation time. To avoid NaN issues inside features
    simulation_time = get_simulation_time(simulation_files, simulation_time=simulation_time)
    # Define empty list containing all pits found in a simulation
    pits = {}
    #  Iterate over all routers
    for router_name in routers_names:
        pits[router_name] = get_pit_sizes(data, router_name, simulation_time)
    # print('pits: {}'.format(pits))
    return pits


def extract_srs_from_simulation_files(simulation_files, simulation_time=300):
    # Extract data from the considered simulation
    data = get_data(simulation_files)
    # Get names of nodes inside a simulation
    consumers_names = get_consumers_names(data)
    # Get simulation time. To avoid NaN issues inside features
    simulation_time = get_simulation_time(simulation_files, simulation_time=simulation_time)
    # Define empty list containing all pits found in a simulation
    srs = {}
    #  Iterate over all routers
    for consumer_name in consumers_names:
        srs[consumer_name] = get_satisfaction_rates(data, consumer_name, simulation_time)
    # print('srs: {}'.format(srs))
    return srs


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
        return min(in_satisfied_interests/max(out_interests, 1), 1)


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
    pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
    return min(pit_size, 1200)


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


def main():
    # Define scenarios for which the distribution plot is required
    download_folder = 'ifa_data'
    # Define scenarios for which the distribution plot is required
    scenario = ['normal', 'existing']
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn']
    # Run distribution plotter
    plot_ifa_effects(download_folder, scenario, topologies)


if __name__ == '__main__':
    main()
