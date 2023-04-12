# Python modules
import random
import pandas as pd
import numpy as np
import math
import os
import glob
import warnings
import networkx as nx
import pickle
import matplotlib.pyplot as plt
# Import modules
from non_ml_baselines.utils import timeit, get_scenario_labels_dict, get_attacker_type_labels_dict, \
    get_topology_labels_dict
from .sample import Sample

warnings.filterwarnings("ignore")


class Extractor():
    def __init__(self, data_dir='ifa_data',
                 scenario='existing',
                 topology='small',
                 n_attackers=None,
                 simulation_time=300,
                 time_att_start=50):
        self.data_dir = data_dir

        self.scenario = scenario
        self.topology = topology
        # assert n_attackers is not None
        self.n_attackers = n_attackers

        self.simulation_time = simulation_time
        self.time_att_start = time_att_start

    @staticmethod
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
                file = Extractor.convert_pit_to_decent_format(file)
            if file_type == 'topology':
                # print('Converting topology file to decent format...')
                file = Extractor.convert_topology_to_decent_format(file)
            # Read csv file
            file_data = pd.read_csv(file, sep='\t', index_col=False)
            # Put data in dictionary
            data[file_type] = file_data

        def refine_routers_names(name):
            try:
                if 'gw-' in name or 'bb-' in name:
                    if 'PIT' in name:
                        return 'PIT_' + name.split('-')[-1]
                    else:
                        return 'Rout' + name.split('-')[-1]
                else:
                    return name
            except TypeError:
                # print(f'\nname: {name}\n')
                return name

        for elm, dtf in data.items():
            # print('elm: {}'.format(elm))
            # print('dtf: {}'.format(dtf))
            if elm != 'topology':
                # print('dtf[\'Node\']: {}'.format(dtf['Node']))
                dtf['Node'] = dtf['Node'].apply(refine_routers_names)
        return data

    @staticmethod
    def reformat_files(files):
        # Iterate over all simulation files and read them
        for file in files:
            # print('file: {}'.format(file))
            # Check file type
            file_type = file.split('/')[-1].split('-')[0]
            if file_type == 'pit':
                Extractor.convert_pit_to_decent_format(file)
            if file_type == 'topology':
                Extractor.convert_topology_to_decent_format(file)
        return

    @staticmethod
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

    @staticmethod
    def remove_topo_data_from_dict(data):
        # print(f'data: {data}')
        data = {key: value for key, value in data.items() if key.split('/')[-1].split('-')[0] != 'topology'}
        # print(f'data: {data}')
        return data

    @staticmethod
    def get_router_names(data):
        # print(f'data: {data}')
        # print(f'data.keys(): {data.keys()}')
        data = Extractor.remove_topo_data_from_dict(data)
        # print(f'data: {data}')
        # print(f'data.keys(): {data.keys()}')
        # Get names of transmitter devices
        routers_names = data['rate']['Node'].unique()
        # print('names: {}'.format(routers_names))
        # Consider routers only
        routers_names = [i for i in routers_names if 'Rout' in i]
        # print('routers_names: {}'.format(routers_names))
        return routers_names

    @staticmethod
    def filter_data_by_time(data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        # Remove topology data from data dictionary
        data = Extractor.remove_topo_data_from_dict(data)
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = data[key][data[key]['Time'] == time]
        return filtered_data

    @staticmethod
    def extract_data_up_to(data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        data = Extractor.remove_topo_data_from_dict(data)
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = data[key][data[key]['Time'] <= time]
        return filtered_data

    @staticmethod
    def get_lines_from_unformatted_topology_file(path_to_file):
        reversed_lines = reversed(list(open(path_to_file)))
        lines_to_keep = []
        for line in reversed_lines:
            if line.rstrip() == '':
                continue
            if line.rstrip()[0] == '#':
                break
            else:
                lines_to_keep.append('\t'.join(line.split()) + '\n')
        lines_to_keep = reversed(lines_to_keep)
        # print('lines_to_keep: {}'.format(lines_to_keep))
        return lines_to_keep

    @staticmethod
    def convert_topology_to_decent_format(file):
        final_lines = Extractor.get_lines_from_unformatted_topology_file(file)
        # Store file with new name
        new_file = os.path.join('/', os.path.join(*file.split('/')[:-1]),
                                'format-{}'.format(file.split('/')[-1]))
        if os.path.exists(new_file):
            os.remove(new_file)
        with open(new_file, 'w') as f:
            f.write('Source\tDestination\n')
            for item in final_lines:
                # Keep only source & destination and remove gw- & bb- from large routers names
                # print('item: {}'.format(item))
                splits = item.split()
                src = splits[0]
                if 'gw-' in src or 'bb-' in src:
                    src = 'Rout' + src.split('-')[-1]
                dst = splits[1]
                if 'gw-' in dst or 'bb-' in dst:
                    dst = 'Rout' + dst.split('-')[-1]
                item = '{}\t{}\n'.format(src, dst)
                # print('formatted item: {}'.format(item))
                f.write(item)
        return new_file

    @staticmethod
    def get_graph_structure(topology_lines_dataframe, routers_names, debug=False):
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
        for index, row in topology_lines_dataframe.iterrows():
            source = row['Source']
            dest = row['Destination']
            if source in routers_names and dest in routers_names:  # source[:4] == 'Rout' and dest[:4] == 'Rout':
                router_links.append([source[4:], dest[4:]])
        # print('router_links: {}'.format(router_links))
        list_of_nodes = list(set([elem for link in router_links for elem in link]))
        for node in routers_names:
            if node.split('Rout')[-1] not in list_of_nodes:
                # print('Node not found is: {}'.format(node))
                pass
        # Use nx to obtain the graph corresponding to the graph
        graph = nx.DiGraph()
        # Build the DODAG graph from nodes and edges lists
        graph.add_nodes_from(list_of_nodes)
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

    @staticmethod
    def get_node_features(whole_data, filtered_data, node_name, routers_names):
        # Get different modes of data
        whole_rate_data = whole_data['rate']
        whole_pit_data = whole_data['pit']
        whole_drop_data = whole_data['drop']
        # Get pit size of router at hand
        router_index = node_name.split('Rout')[-1]
        # Get all interfaces of the current router
        # print('\n\n')
        # print('node_name: {}'.format(node_name))
        # print(rate_data[(rate_data['Node'] == node_name)]['FaceId'])
        # interfaces = list(set(rate_data[(rate_data['Node'] == node_name)]['FaceId'].to_list()))
        # interfaces = [interface for interface in interfaces if interface not in ['-1', -1]]
        interfaces = list(set(whole_rate_data[(whole_rate_data['Time'] == 51) &
                                              (whole_rate_data['Node'] == node_name)]['FaceId'].to_list()))
        interfaces = [interface for interface in interfaces if interface not in ['-1', -1]]

        # Get different modes of data
        rate_data = filtered_data['rate']
        pit_data = filtered_data['pit']
        drop_data = filtered_data['drop']
        # print('interfaces: {}'.format(interfaces))
        features = {interface: {} for interface in interfaces}
        for interface in interfaces:
            for feature in ['in_interests', 'out_interests', 'in_data', 'out_data',
                            'in_nacks', 'out_nacks', 'in_satisfied_interests', 'in_timedout_interests',
                            'out_satisfied_interests', 'out_timedout_interests']:
                features[interface][feature] = Extractor.get_feature(rate_data, node_name, interface, feature)
        # Compute all features used by non ml baselines
        features = Extractor.redistribute_pits(pit_data, router_index, features, interfaces)
        features = Extractor.compute_omega(features, interfaces)
        features = Extractor.compute_por(features, interfaces)
        features = Extractor.compute_per(features, interfaces)
        features = Extractor.compute_isr(features, interfaces)
        features = Extractor.compute_sr(features, interfaces)
        # Return feature for node node_name
        # for key, value in features.items():
        #     print('features of interface {} : {}'.format(key, value))
        nan_count = 0
        for int_key, interface_features in features.items():
            for _, value in interface_features.items():
                if math.isnan(value):
                    nan_count += 1
        if nan_count != 0:
            raise ValueError('Something very wrong! All features are zeros!\n'
                             'Features = {}'.format(features))
        # raise ValueError()
        return features

    @staticmethod
    def get_feature(rate_data, node_name, interface, feature_name):
        feature_name_dict = {'in_interests': 'InInterests',
                             'out_interests': 'OutInterests',
                             'in_data': 'InData',
                             'out_data': 'OutData',
                             'in_nacks': 'InNacks',
                             'out_nacks': 'OutNacks',
                             'in_satisfied_interests': 'InSatisfiedInterests',
                             'in_timedout_interests': 'InTimedOutInterests',
                             'out_satisfied_interests': 'OutSatisfiedInterests',
                             'out_timedout_interests': 'OutTimedOutInterests'}
        feature = rate_data[(rate_data['Node'] == node_name) & (rate_data['FaceId'] == interface) & (
                rate_data['Type'] == feature_name_dict[feature_name])]['PacketRaw']
        # print('node_name: {}'.format(node_name))
        # print('interface: {}'.format(interface))
        # print('feature_name: {}'.format(feature_name))
        # print('{} = {}'.format(feature_name, feature))
        # print(feature)
        # assert len(feature.to_list()) == 1
        try:
            # print('returning {} for router {} and interface {}'.format(feature.to_list()[0], node_name, interface))
            return feature.to_list()[0]
        except IndexError:
            # print('returning {} for router {} and interface {}'.format(0., node_name, interface))
            return 0.

    @staticmethod
    def redistribute_pits(pit_data, router_index, features, interfaces):
        # Compute total pit size and distribute it among interfaces
        total_pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
        # print('total pit size: {}'.format(total_pit_size))
        in_interests_distribution = {interface: features[interface]['in_interests'] for interface in interfaces}
        # print('in_interests_distribution: {}'.format(in_interests_distribution))
        tot_in_interests = sum([value for key, value in in_interests_distribution.items()])
        # print('tot_in_interests: {}'.format(tot_in_interests))
        if tot_in_interests == 0:
            tot_in_interests = 1
        in_interests_distribution = {key: value / float(tot_in_interests) for key, value in
                                     in_interests_distribution.items()}
        # print('in_interests_distribution: {}'.format(in_interests_distribution))
        pit_sizes = {interface: int(in_interests_distribution[interface] * total_pit_size) for interface in interfaces}
        # print('pit_sizes: {}'.format(pit_sizes))
        missing_pits = total_pit_size - sum([val for _, val in pit_sizes.items()])
        all_zeros = all(v == 0 for v in list(in_interests_distribution.values()))
        # print('all_zeros: {}'.format(all_zeros))
        counter = 0
        while counter < missing_pits:
            key, value = random.choice(list(pit_sizes.items()))
            if in_interests_distribution[key] != 0 or all(v == 0 for v in list(in_interests_distribution.values())):
                pit_sizes[key] += 1
                counter += 1
        # print('pit_sizes: {}'.format(pit_sizes))
        # for k in range(missing_pits):
        #     key, value = random.choice(list(pit_sizes.items()))
        #     pit_sizes[key] += 1
        # print('pit_sizes: {}'.format(pit_sizes))
        for interface in interfaces:
            features[interface]['pit_size'] = pit_sizes[interface]
        return features

    @staticmethod
    def compute_omega(features, interfaces):
        # Compute omega as the division between incoming interests and out data
        for interface in interfaces:
            try:
                features[interface]['omega'] = float(features[interface]['in_interests']) / float(
                    features[interface]['out_data'])
            except ZeroDivisionError:
                features[interface]['omega'] = 0.
        return features

    @staticmethod
    def compute_por(features, interfaces):
        # Compute por as pit size divided by 1200
        for interface in interfaces:
            features[interface]['por'] = float(features[interface]['pit_size']) / 1200.
        return features

    @staticmethod
    def compute_per(features, interfaces):
        # Compute per as the division between incoming timed out and (timed out + satisfied) interests
        for interface in interfaces:
            if features[interface]['in_timedout_interests'] == 0. and \
                    features[interface]['in_satisfied_interests'] == 0.:
                features[interface]['per'] = 0.
            else:
                features[interface]['per'] = float(features[interface]['in_timedout_interests']) / float(
                    features[interface]['in_timedout_interests'] + features[interface]['in_satisfied_interests'])
        return features

    @staticmethod
    def compute_isr(features, interfaces):
        # Compute isr as the fraction between incoming timed out and satisfied interests
        for interface in interfaces:
            if features[interface]['in_timedout_interests'] == 0. and \
                    features[interface]['in_satisfied_interests'] == 0.:
                features[interface]['isr'] = 0.
            elif features[interface]['in_timedout_interests'] != 0. and \
                    features[interface]['in_satisfied_interests'] == 0.:
                features[interface]['isr'] = 1.
            else:
                features[interface]['isr'] = float(features[interface]['in_timedout_interests']) / float(
                    features[interface]['in_satisfied_interests'])
        return features

    @staticmethod
    def compute_sr(features, interfaces):
        # Compute sr as the fraction between incoming data and out interests
        for interface in interfaces:
            if features[interface]['out_interests'] == 0.:
                features[interface]['sr'] = 1.
            else:
                features[interface]['sr'] = float(features[interface]['in_data']) / float(
                    features[interface]['out_interests'])
        return features

    def get_all_nodes_features(self, nodes_names, whole_data, filtered_data):
        # Define empty list for nodes features
        nodes_features = {}
        # Iterate over each node and get their features
        # print('nodes_names: {}'.format(nodes_names))
        for node_index, node_name in enumerate(nodes_names):
            # print('node_index: {}'.format(node_index))
            # print('node_name: {}'.format(node_name))
            features = self.get_node_features(whole_data=whole_data,
                                              filtered_data=filtered_data,
                                              node_name=node_name,
                                              routers_names=nodes_names)
            # print('features: {}'.format(features))
            nodes_features[node_name.split('Rout')[-1]] = features
        #     print('nodes_features: {}'.format(nodes_features))
        # print('nodes_features: {}'.format(nodes_features))
        # Return nodes_features
        return nodes_features

    def insert_labels(self, sample, time, topology, scenario, frequency, attackers, n_attackers, sim_id):
        # print('Inserting labels...')
        if scenario != 'normal':
            # CHeck if time of the current window is before or after the attack start time
            attack_is_on = True if time > self.time_att_start else False
        else:
            attack_is_on = False
        # If attack is on set graph label to 1 else to 0
        sample.insert_label(attack_is_on)
        # Append graph label corresponding to the simulation considered
        sample.insert_scenario(get_scenario_labels_dict()[scenario])
        # Set also time for debugging purposes
        sample.insert_time(time)
        # Set also attack frequency for debugging purposes
        if scenario != 'normal':
            sample.insert_frequency(int(frequency))
        else:
            sample.insert_frequency(-1)
        # Set also attack frequency for dataset extraction purposes
        if scenario != 'normal':
            sample.insert_attackers_type(get_attacker_type_labels_dict()[attackers])
        else:
            sample.insert_attackers_type(-1)
        # Set also attack frequency for dataset extraction purposes
        if scenario != 'normal':
            sample.insert_n_attackers(int(n_attackers))
        else:
            sample.insert_n_attackers(-1)
        # Append graph label corresponding to the simulation id
        sample.insert_sim_id(sim_id)
        # Append topology name
        sample.insert_topology_name(topology)
        # Return graph with labels
        return sample

    def get_simulation_time(self, simulation_files):
        # print('simulation_files: {}'.format(simulation_files))
        # Check if simulation has run up until the end or not. To avoid NaN issues inside features
        rate_trace_file = [file for file in simulation_files if 'rate-trace' in file][0]
        last_line_of_rate_trace_file = pd.read_csv(rate_trace_file, sep='\t', index_col=False).iloc[-1]
        simulation_time_from_rate_trace_file = last_line_of_rate_trace_file['Time']
        # Set simulation time depending on the last line of the trace file
        if simulation_time_from_rate_trace_file < self.simulation_time - 1:
            simulation_time = simulation_time_from_rate_trace_file - 1
        else:
            simulation_time = self.simulation_time - 1
        # Double check simulation time from the pit trace file
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

    def extract_graphs_from_simulation_files(self, simulation_files, simulation_index,
                                             total_simulations, debug=False):
        # print('simulation_files: {}'.format(simulation_files))
        # Extract data from the considered simulation
        data = self.get_data(simulation_files)
        # Get names of nodes inside a simulation
        routers_names = self.get_router_names(data)
        # print('routers_names: {}'.format(routers_names))
        # print('len of routers: {}'.format(len(routers_names)))
        # Define start time as one
        start_time = 1
        # Define empty list containing all graphs found in a simulation
        samples = []
        # Get simulation time
        simulation_time = self.get_simulation_time(simulation_files)
        # For each index get the corresponding network traffic window and extract the features in that window
        for time in range(start_time, simulation_time + 1):
            # print('simulation_files[0].split("/")[-6]: {}'.format(simulation_files[0].split("/")[-6]))
            # print('simulation_files[0].split("/")[-3]: {}'.format(simulation_files[0].split("/")[-3]))
            if simulation_files[0].split("/")[-3] == 'normal':
                scenario = 'normal'
            elif simulation_files[0].split("/")[-6] == 'IFA_4_existing':
                scenario = 'existing'
            elif simulation_files[0].split("/")[-6] == 'IFA_4_non_existing':
                scenario = 'non_existing'
            else:
                raise ValueError('Something wrong with scenario extraction from path!')
            if scenario != 'normal':
                # Print info
                frequency = simulation_files[0].split("/")[-3].split('x')[0]
                attackers = simulation_files[0].split("/")[-4].split('_')[0]
                n_attackers = simulation_files[0].split("/")[-2].split('_')[0]
                print("\r| Extracting... |"
                      " Scenario: {} | Topology: {} |"
                      " Attackers selection: {} |"
                      " N attackers: {} |"
                      " Frequency: {} |"
                      " Simulation progress: {}/{} |"
                      " Time steps progress: {}/{} |".format(scenario,
                                                             self.topology,
                                                             attackers,
                                                             n_attackers,
                                                             frequency,
                                                             simulation_index,
                                                             total_simulations,
                                                             time,
                                                             simulation_time),
                      end="\r")
            else:
                frequency = None
                attackers = None
                n_attackers = None
                print(
                    "\r| Extracting... |"
                    " Scenario: {} | Topology: {} |"
                    " Simulation progress: {}/{} |"
                    " Time steps progress: {}/{} |".format(scenario,
                                                           self.topology,
                                                           simulation_index,
                                                           total_simulations,
                                                           time,
                                                           simulation_time),
                    end="\r")
            # if self.scenario == 'existing' and self.topology == 'dfn' and frequency == '32' \
            #         and split == 'train' and simulation_index == 1 and time >= 299:
            #     continue
            # print(f'data: {data}')
            # Get graph of the network during the current time window
            graph = self.get_graph_structure(topology_lines_dataframe=data['topology'],
                                             routers_names=routers_names)
            # Compute features only on current time window
            filtered_data = self.filter_data_by_time(data, time)
            # print('time: {}'.format(time))
            nodes_features = self.get_all_nodes_features(nodes_names=routers_names,
                                                         whole_data=data,
                                                         filtered_data=filtered_data)
            # print('nodes_features: {}'.format(nodes_features))
            # Construct sample starting from nodes features and graph topology
            sample = Sample(routers_feat=nodes_features,
                            graph=graph)
            # Add labels to the graph as graph and nodes attributes
            sample = self.insert_labels(sample,
                                        time=time,
                                        topology=self.topology,
                                        scenario=scenario,
                                        frequency=frequency,
                                        attackers=attackers,
                                        n_attackers=n_attackers,
                                        sim_id=simulation_index)
            # Append the graph for the current time window to the list of graphs
            samples.append(sample)
        # Return the list of pytorch geometric graphs
        for sample in samples:
            if sample.get_label() not in [0, 1]:
                raise ValueError('The sample does not have a label attribute!')
        return samples

    @staticmethod
    def store_samples(samples, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, file_name), 'wb') as file:
            pickle.dump(samples, file)

    @staticmethod
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

    @timeit
    def run(self, downloaded_data_file, raw_dir, sample_file_name):
        # Iterate over train validation and test and get graph samples
        print('Extracting graph data from each simulation. This may take a while...')
        list_of_samples = []
        # Get attackers mode
        att_modes = set([file.split('/')[-4].split('_')[0] for file in downloaded_data_file])
        for att_mode in att_modes:
            att_files = [file for file in downloaded_data_file if file.split('/')[-4].split('_')[0] == att_mode]
            # Iterate over frequencies
            frequencies = np.unique([file.split('/')[-3].split('x')[0] for file in att_files])
            # print('frequencies: {}'.format(frequencies))
            for frequence in frequencies:
                freq_files = [file for file in att_files if file.split('/')[-3].split('x')[0] == frequence]
                # print('freq_files: {}'.format(freq_files))
                # Iterate over number of attackers
                n_atts = set([file.split('/')[-2].split('_')[0] for file in freq_files])
                for n_att in n_atts:
                    n_att_files = [file for file in freq_files if file.split('/')[-2].split('_')[0] == n_att]
                    # Iterating over index of simulations
                    simulation_indices = [1, 2, 3, 4, 5]
                    for s_index, simulation_index in enumerate(simulation_indices):
                        simulation_files = [file for file in n_att_files if
                                            int(file.split('-')[-1].split('.')[0]) == simulation_index]
                        if not simulation_files:
                            continue
                        # Extract graphs from single simulation
                        samples = self.extract_graphs_from_simulation_files(simulation_files=simulation_files,
                                                                            simulation_index=s_index + 1,
                                                                            total_simulations=len(
                                                                                simulation_indices))
                        # Add the graphs to the list of tg_graphs
                        list_of_samples += samples
        # print('list_of_tg_graphs: {}'.format(list_of_tg_graphs))
        # Close info line
        print()
        # Store list of tg graphs in the raw folder of the tg dataset
        self.store_samples(list_of_samples,
                           raw_dir,
                           sample_file_name)
