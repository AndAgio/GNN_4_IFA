# Python modules
import pandas as pd
import numpy as np
import math
import random
import os
import glob
import warnings
import networkx as nx
import torch
import torch_geometric as tg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
# Import modules
from gnn4ifa.utils import timeit, get_scenario_labels_dict, get_attacker_type_labels_dict, get_topology_labels_dict

warnings.filterwarnings("ignore")


class Extractor():
    def __init__(self, data_dir='ifa_data',
                 scenario='existing',
                 topology='small',
                 n_attackers=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 differential=False):
        self.data_dir = data_dir

        self.scenario = scenario
        self.topology = topology
        # assert n_attackers is not None
        self.n_attackers = n_attackers

        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids

        self.simulation_time = simulation_time
        self.time_att_start = time_att_start

        self.differential = differential

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
    def get_node_features(data, node_name, routers_names, mode='array'):
        if mode == 'array':
            features = np.zeros((12), dtype=float)
        elif mode == 'dict':
            features = {}
        else:
            raise ValueError('Invalid mode for extracting node features!')
        # Get different modes of data
        rate_data = data['rate']
        pit_data = data['pit']
        drop_data = data['drop']
        # Get pit size of router at hand
        router_index = node_name.split('Rout')[-1]
        # print('data: {}'.format(data))
        # print('node_name: {}'.format(node_name))
        # print('router_index: {}'.format(router_index))
        # print(pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)])
        # print(pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'])
        list_of_nodes_in_pit = pit_data['Node'].tolist()
        # print('len of list_of_nodes_in_pit: {}'.format(len(list_of_nodes_in_pit)))
        # print('len of list_of_nodes_in_pit without duplicates: {}'.format(len(list(set(list_of_nodes_in_pit)))))
        dupes = [x for n, x in enumerate(list_of_nodes_in_pit) if x in list_of_nodes_in_pit[:n]]
        # print('duplicates: {}'.format(dupes))
        # print('len of duplicates: {}'.format(len(dupes)))
        indices_in_pit = [name.split('_')[-1] for name in list_of_nodes_in_pit]
        indices_in_routers = [name.split('Rout')[-1] for name in routers_names]
        # print('indices_in_pit: {}'.format(indices_in_pit))
        # print('indices_in_routers: {}'.format(indices_in_routers))
        indices_in_pit_but_not_in_topology = list(set(indices_in_pit) - set(indices_in_routers))
        # print('indices_in_pit_but_not_in_topology: {}'.format(indices_in_pit_but_not_in_topology))
        indices_in_topology_but_not_in_pit = list(set(indices_in_routers) - set(indices_in_pit))
        # print('indices_in_topology_but_not_in_pit: {}'.format(indices_in_topology_but_not_in_pit))

        pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
        features[0 if mode == 'array' else 'pit_size'] = pit_size
        # Get drop rate of router at hand
        try:
            drop_rate = drop_data[drop_data['Node'] == node_name]['PacketsRaw'].item()
        except ValueError:
            drop_rate = 0
        # if math.isnan(drop_rate):
        #     print('drop rate: {}'.format(drop_rate))
        #     raise ValueError('NaN found in drop rate!')
        features[1 if mode == 'array' else 'drop_rate'] = drop_rate
        # Get InInterests of router at hand
        in_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InInterests')]['PacketRaw']
        in_interests_list = in_interests.to_list()
        in_interests = sum(i for i in in_interests_list)
        features[2 if mode == 'array' else 'in_interests'] = in_interests
        # Get OutInterests of router at hand
        out_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutInterests')]['PacketRaw']
        out_interests_list = out_interests.to_list()
        out_interests = sum(i for i in out_interests_list)
        features[3 if mode == 'array' else 'out_interests'] = out_interests
        # Get InData of router at hand
        in_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InData')]['PacketRaw']
        in_data_list = in_data.to_list()
        in_data = sum(i for i in in_data)
        features[4 if mode == 'array' else 'in_data'] = in_data
        # Get OutData of router at hand
        out_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutData')]['PacketRaw']
        out_data_list = out_data.to_list()
        out_data = sum(i for i in out_data_list)
        features[5 if mode == 'array' else 'out_data'] = out_data
        # Get InNacks of router at hand
        in_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InNacks')]['PacketRaw']
        in_nacks_list = in_nacks.to_list()
        in_nacks = sum(i for i in in_nacks_list)
        features[6 if mode == 'array' else 'in_nacks'] = in_nacks
        # Get OutNacks of router at hand
        out_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutNacks')]['PacketRaw']
        out_nacks_list = out_nacks.to_list()
        out_nacks = sum(i for i in out_nacks_list)
        features[7 if mode == 'array' else 'out_nacks'] = out_nacks
        # Get InSatisfiedInterests of router at hand
        in_satisfied_interests = \
            rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InSatisfiedInterests')]['PacketRaw']
        in_satisfied_interests_list = in_satisfied_interests.to_list()
        in_satisfied_interests = sum(i for i in in_satisfied_interests)
        features[8 if mode == 'array' else 'in_interests'] = in_satisfied_interests
        # Get InTimedOutInterests of router at hand
        in_timedout_interests = \
            rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InTimedOutInterests')]['PacketRaw']
        in_timedout_interests_list = in_timedout_interests.to_list()
        in_timedout_interests = sum(i for i in in_timedout_interests_list)
        features[9 if mode == 'array' else 'in_timedout_interests'] = in_timedout_interests
        # Get OutSatisfiedInterests of router at hand
        out_satisfied_interests = \
            rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutSatisfiedInterests')]['PacketRaw']
        out_satisfied_interests_list = out_satisfied_interests.to_list()
        out_satisfied_interests = sum(i for i in out_satisfied_interests_list)
        features[10 if mode == 'array' else 'out_satisfied_interests'] = out_satisfied_interests
        # Get OutTimedOutInterests of router at hand
        out_timedout_interests = \
            rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutTimedOutInterests')]['PacketRaw']
        out_timedout_interests_list = out_timedout_interests.to_list()
        out_timedout_interests = sum(i for i in out_timedout_interests_list)
        features[11 if mode == 'array' else 'out_timedout_interests'] = out_timedout_interests
        # Return feature for node node_name
        # print('features: {}'.format(features))
        if mode == 'array':
            # print('np.count_nonzero(features): {}'.format(np.count_nonzero(features)))
            if np.isnan(features).any():
                raise ValueError('Something very wrong! All features are zeros!')
        elif mode == 'dict':
            nan_count = 0
            for key, value in enumerate(features):
                if math.isnan(value):
                    nan_count += 1
            if nan_count != 0:
                raise ValueError('Something very wrong! All features are zeros!\n'
                                 'Features = {}'.format(features))
        else:
            raise ValueError('Invalid mode for extracting node features!')

        return features

    def get_all_nodes_features(self, nodes_names, data):
        # Define empty list for nodes features
        nodes_features = {}
        # Iterate over each node and get their features
        for node_index, node_name in enumerate(nodes_names):
            features = self.get_node_features(data=data,
                                              node_name=node_name,
                                              routers_names=nodes_names)
            nodes_features[node_name.split('Rout')[-1]] = features
        # print('nodes_features shape: {}'.format(nodes_features.shape))
        # Return nodes_features
        return nodes_features

    def insert_labels(self, graph, time, topology, scenario, frequency, attackers, n_attackers):
        # print('Inserting labels...')
        if scenario != 'normal':
            # CHeck if time of the current window is before or after the attack start time
            attack_is_on = True if time > self.time_att_start else False
        else:
            attack_is_on = False
        graph.graph['topology'] = get_topology_labels_dict()[topology]
        # If attack is on set graph label to 1 else to 0
        graph.graph['attack_is_on'] = attack_is_on
        # Append graph label corresponding to the simulation considered
        graph.graph['train_scenario'] = get_scenario_labels_dict()[scenario]
        # Set also time for debugging purposes
        graph.graph['time'] = time
        # Set also attack frequency for debugging purposes
        if scenario != 'normal':
            graph.graph['frequency'] = int(frequency)
        else:
            graph.graph['frequency'] = -1
        # Set also attack frequency for dataset extraction purposes
        if scenario != 'normal':
            graph.graph['attackers_type'] = get_attacker_type_labels_dict()[attackers]
        else:
            graph.graph['attackers_type'] = -1
        # Set also attack frequency for dataset extraction purposes
        if scenario != 'normal':
            graph.graph['n_attackers'] = int(n_attackers)
        else:
            graph.graph['n_attackers'] = -1
        # Return graph with labels
        return graph

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
                                             total_simulations, split, debug=False):
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
        tg_graphs = []
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
                print(
                    "\r| Extracting {} split... |"
                    " Scenario: {} | Topology: {} |"
                    " Attackers selection: {} |"
                    " N attackers: {} |"
                    " Frequency: {} |"
                    " Simulation progress: {}/{} |"
                    " Time steps progress: {}/{} |".format(split,
                                                           scenario,
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
                    "\r| Extracting {} split... |"
                    " Scenario: {} | Topology: {} |"
                    " Simulation progress: {}/{} |"
                    " Time steps progress: {}/{} |".format(split,
                                                           scenario,
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
            if self.differential:
                # If differential is required compute difference between current time window and previous time window
                if time == start_time:
                    continue
                filtered_data = self.filter_data_by_time(data, time - 1)
                nodes_features_pre = self.get_all_nodes_features(nodes_names=routers_names,
                                                                 data=filtered_data)
                filtered_data = self.filter_data_by_time(data, time)
                nodes_features_post = self.get_all_nodes_features(nodes_names=routers_names,
                                                                  data=filtered_data)
                nodes_features = {key: nodes_features_post[key] - nodes_features_pre[key] for key in
                                  nodes_features_post.keys()}
            else:
                # If differential is not required compute features only on current time window
                filtered_data = self.filter_data_by_time(data, time)
                nodes_features = self.get_all_nodes_features(nodes_names=routers_names,
                                                             data=filtered_data)
            # Add nodes features to graph
            # print('graph.nodes: {}'.format(graph.nodes))
            # print('nodes_features: {}'.format(nodes_features))
            for node_name in graph.nodes:
                # print('node_name: {}'.format(node_name))
                # print('graph.nodes[node_name]: {}'.format(graph.nodes[node_name]))
                # print('nodes_features[node_name]: {}'.format(nodes_features[node_name]))
                graph.nodes[node_name]['x'] = nodes_features[node_name]
            # Debugging purposes
            # print('graph.nodes.data(): {}'.format(graph.nodes.data()))
            # print('\n\n\nnumber of routers: {}\n\n\n'.format(len(graph.nodes.data())))
            # Add labels to the graph as graph and nodes attributes
            graph = self.insert_labels(graph,
                                       time=time,
                                       topology=self.topology,
                                       scenario=scenario,
                                       frequency=frequency,
                                       attackers=attackers,
                                       n_attackers=n_attackers)
            # Debugging purposes
            if debug:
                print('graph.graph: {}'.format(graph.graph))
                print('graph.nodes.data(): {}'.format(graph.nodes.data()))
                print('graph.edges.data(): {}'.format(graph.edges.data()))
                nx.draw(graph, with_labels=True)
                plt.show()
            # Convert networkx graph into torch_geometric
            tg_graph = tg.utils.from_networkx(graph)
            # Add graph labels to the tg_graph
            for graph_label_name, graph_label_value in graph.graph.items():
                torch.tensor(graph_label_value, dtype=torch.int)
                tg_graph[graph_label_name] = torch.tensor(graph_label_value, dtype=torch.int)
            # print('tg_graph: {}'.format(tg_graph))
            # print('tg_graph.x: {}'.format(tg_graph.x))
            # print('tg_graph.edge_index: {}'.format(tg_graph.edge_index))
            # Append the graph for the current time window to the list of graphs
            tg_graphs.append(tg_graph)
        # Return the list of pytorch geometric graphs
        for sample in tg_graphs:
            try:
                sample.attack_is_on
            except KeyError:
                raise ValueError('The sample {} does not have attack_is_on attribute!'.format(sample))
        # print('tg_graphs: {}'.format(tg_graphs))
        return tg_graphs

    @staticmethod
    def store_tg_data_raw(raw_data, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(raw_data,
                   os.path.join(folder, file_name))

    def split_files(self, files):
        # Split files depending on the train ids
        # print('files: {}'.format(files))
        # for file in files:
        #     try:
        #         int(file.split('-')[-1].split('.')[0])
        #     except ValueError:
        #         print('File raising error is: {}'.format(file))
        print('self.train_sim_ids: {}'.format(self.train_sim_ids))
        train_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.train_sim_ids]
        # print('train_files: {}'.format(train_files))
        val_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.val_sim_ids]
        # print('val_files: {}'.format(val_files))
        test_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.test_sim_ids]
        # print('test_files: {}'.format(test_files))
        return train_files, val_files, test_files

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
    def run(self, downloaded_data_file, raw_dir, raw_file_names, split_mode='file_ids'):
        # print('\nraw_file_names: {}\n'.format(raw_file_names))
        # downloaded_data_file = Extractor.rename_topology_files(downloaded_data_file)
        if split_mode == 'file_ids' or self.differential:
            # Split the received files into train, validation and test
            files_lists = self.split_files(downloaded_data_file)
            # Iterate over train validation and test and get graph samples
            print('Extracting graph data from each simulation of each split. This may take a while...')
            for index, files in enumerate(files_lists):
                list_of_tg_graphs = []
                # Get attackers mode
                att_modes = set([file.split('/')[-4].split('_')[0] for file in files])
                for att_mode in att_modes:
                    att_files = [file for file in files if file.split('/')[-4].split('_')[0] == att_mode]
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
                            if index == 0:
                                split = 'train'
                                simulation_indices = self.train_sim_ids
                            elif index == 1:
                                split = 'val'
                                simulation_indices = self.val_sim_ids
                            elif index == 2:
                                split = 'test'
                                simulation_indices = self.test_sim_ids
                            else:
                                raise ValueError('Something went wrong with simulation indices')
                            for s_index, simulation_index in enumerate(simulation_indices):
                                simulation_files = [file for file in n_att_files if
                                                    int(file.split('-')[-1].split('.')[0]) == simulation_index]
                                # print('simulation files: {}'.format(simulation_files))
                                if not simulation_files:
                                    continue
                                # Extract graphs from single simulation
                                tg_graphs = self.extract_graphs_from_simulation_files(simulation_files=simulation_files,
                                                                                      simulation_index=s_index + 1,
                                                                                      total_simulations=len(
                                                                                          simulation_indices),
                                                                                      split=split)
                                # Add the graphs to the list of tg_graphs
                                list_of_tg_graphs += tg_graphs
                # print('list_of_tg_graphs: {}'.format(list_of_tg_graphs))
                # Close info line
                print()
                # Store list of tg graphs in the raw folder of the tg dataset
                self.store_tg_data_raw(list_of_tg_graphs,
                                       raw_dir,
                                       raw_file_names[index])
        elif split_mode == 'percentage':
            print('Trying to load all graphs converted to tg samples...')
            try:
                uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
                pickles_path = uppath(raw_dir, 1)
                print('Looking for pickle file containing '
                      'all graphs at {}'.format(os.path.join(pickles_path,
                                                             'pickles_checkpoints',
                                                             'all_tg_graphs_list_diff_{}.pkl'.format(
                                                                 self.differential))))
                with open(os.path.join(pickles_path,
                                       'pickles_checkpoints',
                                       'all_tg_graphs_list_diff_{}.pkl'.format(self.differential)), 'rb') as f:
                    print('Pickle file containing all graphs found. Loading it...')
                    list_of_tg_graphs = pickle.load(f)
            except:
                print('Pickle file containing all graphs not found!')
                # Split the received files into train, validation and test
                files = downloaded_data_file
                # Iterate over train validation and test and get graph samples
                print('Extracting graph data from each simulation. This may take a while...')
                list_of_tg_graphs = []
                # counter = 0
                # Get attackers mode
                att_modes = set([file.split('/')[-4].split('_')[0] for file in files])
                if 'fixed' in att_modes:
                    att_modes.remove('fixed')
                print('\n\natt_modes: {}\n\n'.format(att_modes))
                # att_modes = ['variable']
                # print('att_modes = {}'.format(att_modes))
                for att_mode in att_modes:
                    att_files = [file for file in files if file.split('/')[-4].split('_')[0] == att_mode]
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
                                # print('simulation files: {}'.format(simulation_files))
                                if not simulation_files:
                                    continue
                                # Extract graphs from single simulation
                                tg_graphs = self.extract_graphs_from_simulation_files(simulation_files=simulation_files,
                                                                                      simulation_index=s_index + 1,
                                                                                      total_simulations=len(
                                                                                          simulation_indices),
                                                                                      split='all')
                                # Add the graphs to the list of tg_graphs
                                list_of_tg_graphs += tg_graphs
                    #             counter += 1
                    #             if counter >= 2:
                    #                 break
                    #         if counter >= 2:
                    #             break
                    #     if counter >= 2:
                    #         break
                    # if counter >= 2:
                    #     break
                    # print('list_of_tg_graphs: {}'.format(list_of_tg_graphs))
                    # Close info line
                print()
                # Store list of tg graphs in the raw folder of the tg dataset
                uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
                pickles_path = uppath(raw_dir, 1)
                if not os.path.exists(os.path.join(pickles_path, 'pickles_checkpoints')):
                    os.makedirs(os.path.join(pickles_path, 'pickles_checkpoints'))
                print('Dumping list of to {}'.format(os.path.join(pickles_path,
                                                                  'pickles_checkpoints',
                                                                  'all_tg_graphs_list_diff_{}.pkl'.format(
                                                                      self.differential))))
                with open(os.path.join(pickles_path,
                                       'pickles_checkpoints',
                                       'all_tg_graphs_list_diff_{}.pkl'.format(self.differential)), 'wb') as f:
                    pickle.dump(list_of_tg_graphs, f)
            with open(os.path.join(pickles_path,
                                   'pickles_checkpoints',
                                   'all_tg_graphs_list_diff_{}.pkl'.format(self.differential)), 'rb') as f:
                print('Pickle file containing all graphs found. Loading it...')
                list_of_tg_graphs = pickle.load(f)
            print('Splitting graphs into sets depending on given percentages. This may take a while...')
            # Split dataset into train, validation and test
            random.shuffle(list_of_tg_graphs)
            # train_pct_index = int(self.train_sim_ids * len(list_of_tg_graphs))
            # train_data = list_of_tg_graphs[:train_pct_index]
            # rest_data = list_of_tg_graphs[train_pct_index:]
            train_data, rest_data = train_test_split(list_of_tg_graphs, train_size=self.train_sim_ids, shuffle=False)
            # print('train_data: {}'.format(train_data))
            val_data_percentage = self.val_sim_ids / (1 - self.train_sim_ids)
            # print('self.train_sim_ids: {}'.format(self.train_sim_ids))
            # print('self.val_sim_ids: {}'.format(self.val_sim_ids))
            # print('self.test_sim_ids: {}'.format(self.test_sim_ids))
            # print('val_data_percentage: {}'.format(val_data_percentage))
            # val_pct_index = int(val_data_percentage * len(rest_data))
            # validation_data = rest_data[:val_pct_index]
            # test_data = rest_data[val_pct_index:]
            # list_of_lists_of_tg_graphs = [train_data, validation_data, test_data]
            validation_data, test_data = train_test_split(rest_data, train_size=val_data_percentage, shuffle=False)
            list_of_lists_of_tg_graphs = [train_data, validation_data, test_data]
            for index in range(3):
                print('Trying to store split at index {}...'.format(index))
                self.store_tg_data_raw(list_of_lists_of_tg_graphs[index],
                                       raw_dir,
                                       raw_file_names[index])
                print('Storing completed successfully!')
            # print('Trying to store training split...')
            # self.store_tg_data_raw(train_data,
            #                        raw_dir,
            #                        raw_file_names[0])
            # print('Storing completed successfully!')
            # print('Trying to store validation split...')
            # self.store_tg_data_raw(validation_data,
            #                        raw_dir,
            #                        raw_file_names[1])
            # print('Storing completed successfully!')
            # print('Trying to store test split...')
            # self.store_tg_data_raw(test_data,
            #                        raw_dir,
            #                        raw_file_names[2])
            # print('Storing completed successfully!')
        else:
            raise ValueError('Split mode \'{}\' not available!'.format(split_mode))
