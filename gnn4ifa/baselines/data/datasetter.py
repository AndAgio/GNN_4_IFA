# Python modules
import pandas as pd
import numpy as np
import math
import os
import glob
import warnings
import json
# Import modules
from baselines.utils import timeit, get_scenario_labels_dict, get_attacker_type_labels_dict, get_topology_labels_dict

warnings.filterwarnings("ignore")


class Datasetter:
    def __init__(self,
                 data_dir='ifa_data',
                 scenario='existing',
                 topology='small',
                 n_attackers=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 mode='single',
                 selected_features='all',
                 out_file='baseline_dataset'):
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

        self.mode = mode
        if selected_features == 'all' or selected_features == ['all']:
            self.selected_features = ['pit_size',
                                      'drop_rate',
                                      'in_interests',
                                      'out_interests',
                                      'in_data',
                                      'out_data',
                                      'in_nacks',
                                      'out_nacks',
                                      'in_satisfied_interests',
                                      'in_timedout_interests',
                                      'out_satisfied_interests',
                                      'out_timedout_interests']
        else:
            self.selected_features = selected_features
        self.out_file = out_file

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
                file = Datasetter.convert_pit_to_decent_format(file)
            if file_type == 'topology':
                # print('Converting topology file to decent format...')
                file = Datasetter.convert_topology_to_decent_format(file)
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
                Datasetter.convert_pit_to_decent_format(file)
            if file_type == 'topology':
                Datasetter.convert_topology_to_decent_format(file)
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
        data = Datasetter.remove_topo_data_from_dict(data)
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
        data = Datasetter.remove_topo_data_from_dict(data)
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = data[key][data[key]['Time'] == time]
        return filtered_data

    @staticmethod
    def extract_data_up_to(data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        data = Datasetter.remove_topo_data_from_dict(data)
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
        final_lines = Datasetter.get_lines_from_unformatted_topology_file(file)
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

    def get_node_features(self, data, node_name, routers_names):
        features = {feature: None for feature in self.selected_features}
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

        if 'pit_size' in self.selected_features:
            pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
            features['pit_size'] = pit_size
        if 'drop_rate' in self.selected_features:
            # Get drop rate of router at hand
            try:
                drop_rate = drop_data[drop_data['Node'] == node_name]['PacketsRaw'].item()
            except ValueError:
                drop_rate = 0
            features['drop_rate'] = drop_rate
        if 'in_interests' in self.selected_features:
            # Get InInterests of router at hand
            in_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InInterests')][
                'PacketRaw']
            in_interests_list = in_interests.to_list()
            in_interests = sum(i for i in in_interests_list)
            features['in_interests'] = in_interests
        if 'out_interests' in self.selected_features:
            # Get OutInterests of router at hand
            out_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutInterests')][
                'PacketRaw']
            out_interests_list = out_interests.to_list()
            out_interests = sum(i for i in out_interests_list)
            features['out_interests'] = out_interests
        if 'in_data' in self.selected_features:
            # Get InData of router at hand
            in_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InData')]['PacketRaw']
            in_data_list = in_data.to_list()
            in_data = sum(i for i in in_data)
            features['in_data'] = in_data
        if 'out_data' in self.selected_features:
            # Get OutData of router at hand
            out_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutData')]['PacketRaw']
            out_data_list = out_data.to_list()
            out_data = sum(i for i in out_data_list)
            features['out_data'] = out_data
        if 'in_nacks' in self.selected_features:
            # Get InNacks of router at hand
            in_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InNacks')]['PacketRaw']
            in_nacks_list = in_nacks.to_list()
            in_nacks = sum(i for i in in_nacks_list)
            features['in_nacks'] = in_nacks
        if 'out_nacks' in self.selected_features:
            # Get OutNacks of router at hand
            out_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutNacks')]['PacketRaw']
            out_nacks_list = out_nacks.to_list()
            out_nacks = sum(i for i in out_nacks_list)
            features['out_nacks'] = out_nacks
        if 'in_satisfied_interests' in self.selected_features:
            # Get InSatisfiedInterests of router at hand
            in_satisfied_interests = \
                rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InSatisfiedInterests')]['PacketRaw']
            in_satisfied_interests_list = in_satisfied_interests.to_list()
            in_satisfied_interests = sum(i for i in in_satisfied_interests)
            features['in_satisfied_interests'] = in_satisfied_interests
        if 'in_timedout_interests' in self.selected_features:
            # Get InTimedOutInterests of router at hand
            in_timedout_interests = \
                rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InTimedOutInterests')]['PacketRaw']
            in_timedout_interests_list = in_timedout_interests.to_list()
            in_timedout_interests = sum(i for i in in_timedout_interests_list)
            features['in_timedout_interests'] = in_timedout_interests
        if 'out_satisfied_interests' in self.selected_features:
            # Get OutSatisfiedInterests of router at hand
            out_satisfied_interests = \
                rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutSatisfiedInterests')][
                    'PacketRaw']
            out_satisfied_interests_list = out_satisfied_interests.to_list()
            out_satisfied_interests = sum(i for i in out_satisfied_interests_list)
            features['out_satisfied_interests'] = out_satisfied_interests
        if 'out_timedout_interests' in self.selected_features:
            # Get OutTimedOutInterests of router at hand
            out_timedout_interests = \
                rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutTimedOutInterests')]['PacketRaw']
            out_timedout_interests_list = out_timedout_interests.to_list()
            out_timedout_interests = sum(i for i in out_timedout_interests_list)
            features['out_timedout_interests'] = out_timedout_interests
        if 'pit_usage' in self.selected_features:
            pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
            features['pit_usage'] = pit_size/1200.0
        if 'satisfaction_rate' in self.selected_features:
            # Get Satisfaction rate of router at hand
            in_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InData')]['PacketRaw']
            in_data_list = in_data.to_list()
            in_data = sum(i for i in in_data)
            in_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InInterests')][
                'PacketRaw']
            in_interests_list = in_interests.to_list()
            in_interests = sum(i for i in in_interests_list)
            try:
                features['satisfaction_rate'] = float(in_data/in_interests)
            except:
                features['satisfaction_rate'] = 0.
        # Return feature for node node_name
        nan_count = 0
        for key, value in features.items():
            if math.isnan(value):
                nan_count += 1
        if nan_count != 0:
            raise ValueError('Something very wrong! Some features leads to NaN errors!\n'
                             'Features = {}'.format(features))
        return features

    def get_all_nodes_features(self, nodes_names, data):
        # Define empty list for nodes features
        nodes_features = self.build_empty_data_dict()
        # Iterate over each node and get their features
        for node_index, node_name in enumerate(nodes_names):
            features = self.get_node_features(data=data,
                                              node_name=node_name,
                                              routers_names=nodes_names)
            if self.mode == 'single':
                nodes_features['Router_id'].append(node_name)
                for feat_name, feat_value in features.items():
                    nodes_features[feat_name].append(feat_value)
            elif self.mode == 'avg':
                for feat_name, feat_value in features.items():
                    try:
                        nodes_features[feat_name] += feat_value
                    except:
                        nodes_features[feat_name] = feat_value
            elif self.mode == 'cat':
                nodes_features['feat_vector'].append(list(features.values()))
            else:
                raise ValueError('Unknown mode: {} for Datasetter class!'.format(self.mode))
        if self.mode == 'avg':
            for feat_name, feat_value in features.items():
                nodes_features[feat_name] /= (node_index + 1)
        elif self.mode == 'cat':
            nodes_features['feat_vector'] = [elem for feat_vector in nodes_features['feat_vector'] for elem in
                                             feat_vector]
        # print('nodes_features shape: {}'.format(nodes_features.shape))
        # print('nodes_features: {}'.format(nodes_features))
        # Return nodes_features
        return nodes_features

    def insert_labels(self, nodes_data, n_cycles, time, topology, scenario, frequency, attackers, n_attackers):
        # print('Inserting labels...')
        for _ in range(n_cycles):
            if scenario != 'normal':
                # CHeck if time of the current window is before or after the attack start time
                attack_is_on = True if time > self.time_att_start else False
            else:
                attack_is_on = False
            if self.mode == 'single':
                nodes_data['topology'].append(get_topology_labels_dict()[topology])
                # If attack is on set graph label to 1 else to 0
                nodes_data['attack_is_on'].append(attack_is_on)
                # Append graph label corresponding to the simulation considered
                nodes_data['train_scenario'].append(get_scenario_labels_dict()[scenario])
                # Set also time for debugging purposes
                nodes_data['time'].append(time)
                # Set also attack frequency for debugging purposes
                if scenario != 'normal':
                    nodes_data['frequency'].append(int(frequency))
                else:
                    nodes_data['frequency'].append(-1)
                # Set also attack frequency for dataset extraction purposes
                if scenario != 'normal':
                    nodes_data['attackers_type'].append(get_attacker_type_labels_dict()[attackers])
                else:
                    nodes_data['attackers_type'].append(-1)
                # Set also attack frequency for dataset extraction purposes
                if scenario != 'normal':
                    nodes_data['n_attackers'].append(int(n_attackers))
                else:
                    nodes_data['n_attackers'].append(-1)
            else:
                nodes_data['topology'] = get_topology_labels_dict()[topology]
                # If attack is on set graph label to 1 else to 0
                nodes_data['attack_is_on'] = attack_is_on
                # Append graph label corresponding to the simulation considered
                nodes_data['train_scenario'] = get_scenario_labels_dict()[scenario]
                # Set also time for debugging purposes
                nodes_data['time'] = time
                # Set also attack frequency for debugging purposes
                if scenario != 'normal':
                    nodes_data['frequency'] = int(frequency)
                else:
                    nodes_data['frequency'] = -1
                # Set also attack frequency for dataset extraction purposes
                if scenario != 'normal':
                    nodes_data['attackers_type'] = get_attacker_type_labels_dict()[attackers]
                else:
                    nodes_data['attackers_type'] = -1
                # Set also attack frequency for dataset extraction purposes
                if scenario != 'normal':
                    nodes_data['n_attackers'] = int(n_attackers)
                else:
                    nodes_data['n_attackers'] = -1
        # Return graph with labels
        return nodes_data

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

    def extract_data_dict_from_simulation_files(self, data_dict, simulation_files, simulation_index,
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
            # Compute features only on current time window
            filtered_data = self.filter_data_by_time(data, time)
            nodes_data = self.get_all_nodes_features(nodes_names=routers_names,
                                                     data=filtered_data)
            # Add labels to the graph as graph and nodes attributes
            nodes_data = self.insert_labels(nodes_data,
                                            n_cycles=1 if self.mode in ['cat', 'avg'] else len(routers_names),
                                            time=time,
                                            topology=self.topology,
                                            scenario=scenario,
                                            frequency=frequency,
                                            attackers=attackers,
                                            n_attackers=n_attackers)
            # Debugging purposes
            if debug:
                print('nodes_data: {}'.format(nodes_data))
            # Append the graph for the current time window to the list of graphs
            for key, value in nodes_data.items():
                try:
                    data_dict[key].append(value)
                except:
                    data_dict[key] = [data_dict[key]] + [value]
        # Debugging purposes
        if debug:
            print('data_dict: {}'.format(data_dict))
            print('feat length: {} labels length: {}'.format(len(data_dict['out_nacks']),
                                                             len(data_dict['time'])))
        return data_dict

    @staticmethod
    def store_json(raw_data, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        print('Storing json to {}\n'.format(os.path.join(folder, '{}.json'.format(file_name))))
        with open(os.path.join(folder, '{}.json'.format(file_name)), 'w') as file:
            json.dump(raw_data, file)

    def split_files(self, files):
        # Split files depending on the train ids
        # print('files: {}'.format(files))
        # for file in files:
        #     try:
        #         int(file.split('-')[-1].split('.')[0])
        #     except ValueError:
        #         print('File raising error is: {}'.format(file))
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

    def build_empty_data_dict(self):
        if self.mode == 'single':
            data = {'Router_id': []}
            for feature in self.selected_features:
                data[feature] = []
        elif self.mode == 'avg':
            data = {}
            for feature in self.selected_features:
                data[feature] = []
        elif self.mode == 'cat':
            data = {'feat_vector': []}
        else:
            raise ValueError('Mode {} is not available for Datasetter class!'.format(self.mode))
        for label in ['topology', 'attack_is_on', 'train_scenario',
                      'time', 'frequency', 'attackers_type', 'n_attackers']:
            data[label] = []
        return data

    def read_files(self):
        # Import stored dictionary of data
        if self.scenario == 'all':
            dwn_dir = os.path.join(self.data_dir, 'IFA_4_existing', '{}_topology'.format(self.topology) \
                if self.topology != 'dfn' else '{}_topology'.format(self.topology.upper()))
            file_names = glob.glob(os.path.join(dwn_dir, '*', '*', '*', '*.txt'))
            dwn_dir = os.path.join(self.data_dir, 'normal', '{}_topology'.format(self.topology) \
                if self.topology != 'dfn' else '{}_topology'.format(self.topology.upper()))
            file_names += glob.glob(os.path.join(dwn_dir, '*.txt'))
        elif self.scenario != 'normal':
            # print('self.download_dir: {}'.format(self.download_dir))
            file_names = glob.glob(os.path.join(os.path.join(self.data_dir,
                                                             'IFA_4_{}'.format(
                                                                 self.scenario) if self.scenario != 'normal' else self.scenario,
                                                             '{}_topology'.format(
                                                                 self.topology) if self.topology != 'dfn' else '{}_topology'.format(
                                                                 self.topology.upper())), '*', '*', '*', '*.txt'))
        else:
            file_names = glob.glob(os.path.join(os.path.join(self.data_dir,
                                                             'IFA_4_{}'.format(
                                                                 self.scenario) if self.scenario != 'normal' else self.scenario,
                                                             '{}_topology'.format(
                                                                 self.topology) if self.topology != 'dfn' else '{}_topology'.format(
                                                                 self.topology.upper())), '*.txt'))
        return file_names

    def check_file_available(self, split):
        folder = os.path.join('/' + os.path.join(*self.out_file.split('/')[:-1]), split)
        file_name = self.out_file.split('/')[-1] + '_{}_{}_{}'.format(self.mode,
                                                                      self.topology,
                                                                      self.train_sim_ids if split == 'train' else self.val_sim_ids if split == 'val' else self.test_sim_ids)
        file = os.path.join(folder, '{}.json'.format(file_name))
        print('Looking for file {} containing formatted dataset...'.format(file))
        if not os.path.exists(file):
            print('File {} not found, extracting it...'.format(file))
            return False
        print('File {} found, using it...'.format(file))
        return True

    @timeit
    def run(self, debug=False):
        # formatted_files_available = self.check_file_available()
        # print('\nraw_file_names: {}\n'.format(raw_file_names))
        # downloaded_data_file = Datasetter.rename_topology_files(downloaded_data_file)
        downloaded_data_file = self.read_files()
        # print('downloaded_data_file: {}'.format(downloaded_data_file))
        # Split the received files into train, validation and test
        files_lists = self.split_files(downloaded_data_file)
        # Iterate over train validation and test and get graph samples
        print('Extracting data from each simulation of each split. This may take a while...')
        for index, files in enumerate(files_lists):
            data_dict = self.build_empty_data_dict()
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
            if not self.check_file_available(split):
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
                            for s_index, simulation_index in enumerate(simulation_indices):
                                simulation_files = [file for file in n_att_files if
                                                    int(file.split('-')[-1].split('.')[0]) == simulation_index]
                                # print('simulation files: {}'.format(simulation_files))
                                if not simulation_files:
                                    continue
                                # Extract graphs from single simulation
                                data_dict = self.extract_data_dict_from_simulation_files(data_dict=data_dict,
                                                                                         simulation_files=simulation_files,
                                                                                         simulation_index=s_index + 1,
                                                                                         total_simulations=len(
                                                                                             simulation_indices),
                                                                                         split=split)
                # Close info line
                print()
                if debug:
                    for key, values in data_dict.items():
                        # if key in ['out_nacks', 'time']:
                        #     print('key: {} -> values: {}'.format(key, values))
                        print('key: {} -> n values: {}'.format(key, len(values)))
                # Convert dictionary to dataframe
                dataframe = pd.DataFrame.from_dict(data_dict)
                # Store list of tg graphs in the raw folder of the tg dataset
                folder = '/' + os.path.join(*self.out_file.split('/')[:-1])
                # print('split: {}'.format(split))
                folder = os.path.join(folder, split)
                file_name = self.out_file.split('/')[-1] + '_{}_{}_{}'.format(self.mode,
                                                                              self.topology,
                                                                              self.train_sim_ids if split == 'train' else self.val_sim_ids if split == 'val' else self.test_sim_ids)
                self.store_json(data_dict,
                                folder=folder,
                                file_name=file_name)

    def read_split(self, split):
        folder = '/' + os.path.join(*self.out_file.split('/')[:-1])
        folder = os.path.join(folder, split)
        file_name = self.out_file.split('/')[-1] + '_{}_{}_{}'.format(self.mode,
                                                                      self.topology,
                                                                      self.train_sim_ids if split == 'train' else self.val_sim_ids if split == 'val' else self.test_sim_ids)
        data_set = pd.read_json(os.path.join(folder, '{}.json'.format(file_name)))
        # print('dataset for split {} is: {}'.format(data_set, split))
        return data_set
