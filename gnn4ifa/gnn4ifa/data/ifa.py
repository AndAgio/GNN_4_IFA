import math
import os
import random
import shutil
import zipfile
import time
import glob
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
# Import modules
from gnn4ifa.utils import download_file_from_google_drive
from .extractor import Extractor


class IfaDataset(InMemoryDataset):
    def __init__(self,
                 root='ifa_data_tg',
                 download_folder='ifa_data',
                 transform=None,
                 pre_transform=None,
                 scenario='existing',
                 topology='small',
                 n_attackers=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 train_freq=0.7,
                 val_freq=0.15,
                 test_freq=0.15,
                 split_mode='file_ids',
                 simulation_time=300,
                 time_att_start=50,
                 differential=False,
                 split='train'):
        print(f'Setup:\n'
              f'root={root}\n'
              f'download_folder={download_folder}\n'
              f'transform={transform}\n'
              f'pre_transform={pre_transform}\n'
              f'scenario={scenario}\n'
              f'topology={topology}\n'
              f'n_attackers={n_attackers}\n'
              f'train_sim_ids={train_sim_ids}\n'
              f'val_sim_ids={val_sim_ids}\n'
              f'test_sim_ids={test_sim_ids}\n'
              f'train_freq={train_freq}\n'
              f'val_freq={val_freq}\n'
              f'test_freq={test_freq}\n'
              f'split_mode={split_mode}\n'
              f'simulation_time={simulation_time}\n'
              f'time_att_start={time_att_start}\n'
              f'differential={differential}\n'
              f'split={split}\n')
        self.download_folder = download_folder
        assert scenario in ['existing', 'non_existing', 'normal', 'all']
        self.scenario = scenario
        assert topology in ['small', 'dfn', 'large']
        self.topology = topology
        # assert n_attackers is not None
        self.n_attackers = n_attackers
        # for train_sim_id in train_sim_ids:
        #     assert 1 <= train_sim_id <= 5
        # for val_sim_id in val_sim_ids:
        #     assert 1 <= val_sim_id <= 5
        # for test_sim_id in test_sim_ids:
        #     assert 1 <= test_sim_id <= 5
        # assert set(train_sim_ids + val_sim_ids + test_sim_ids) == {1, 2, 3, 4, 5}
        self.split_mode = split_mode
        if self.split_mode == 'percentage' and not train_freq + val_freq + test_freq == 1:
            print('Split percentages do not sum to 1! Fixing val and test split w.r.t. train split!')
            val_freq = (1 - train_freq) / 2.
            test_freq = (1 - train_freq) / 2.
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.train_freq = train_freq
        self.val_freq = val_freq
        self.test_freq = test_freq
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        self.differential = differential
        self.root = root
        super(IfaDataset, self).__init__(self.root, transform, pre_transform)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")
        print('Loading data from {}...'.format(path))
        self.data, self.slices = torch.load(path)

    @property
    def download_dir(self) -> str:
        return os.path.join(self.download_folder,
                            'IFA_4_{}'.format(self.scenario) if self.scenario != 'normal' else self.scenario,
                            '{}_topology'.format(self.topology) if self.topology != 'dfn' else '{}_topology'.format(
                                self.topology.upper()))

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.scenario, self.topology, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.scenario, self.topology, 'processed')

    @property
    def download_file_names(self):
        # if self.scenario == 'existing' and self.topology == 'dfn':
        #     frequencies = ['4x', '8x', '16x', '32x']
        # elif self.scenario == 'existing' and self.topology == 'small':
        #     frequencies = ['4x', '8x', '16x', '32x', '64x']
        # elif self.scenario == 'non_existing' and self.topology == 'dfn':
        #     raise FileNotFoundError(
        #         'Scenario {} and train_topology {} are incompatible at the moment'.format(self.scenario,
        #                                                                                   self.topology))
        # elif self.scenario == 'non_existing' and self.topology == 'small':
        #     frequencies = ['16x', '32x', '64x', '128x', '256x']
        # elif self.scenario == 'normal':
        #     frequencies = None
        # elif self.scenario == 'all':
        #     frequencies = None
        # else:
        #     raise ValueError('Something wrong with train_scenario {} and train_topology {}'.format(self.scenario,
        #                                                                                            self.topology))
        # # Define files that should be available as raw in the dataset
        # names = ['drop-trace', 'pit-size', 'rate-trace']
        # if frequencies:
        #     file_names = ['{}/{}-{}.txt'.format(freq, name, index) for freq in frequencies for name in names for index
        #                   in range(1, 6)]
        # else:
        #     file_names = ['{}-{}.txt'.format(name, index) for name in names for index in range(1, 6)]
        file_names = ['filefilefile']
        return file_names

    @property
    def raw_file_names(self):
        if self.split_mode == 'file_ids' or self.differential:
            return ['train_{}_diff_{}_data.pt'.format(self.train_sim_ids,
                                                      self.differential),
                    'val_{}_diff_{}_data.pt'.format(self.val_sim_ids,
                                                    self.differential),
                    'test_{}_diff_{}_data.pt'.format(self.test_sim_ids,
                                                     self.differential)]
        elif self.split_mode == 'percentage':
            return ['train_{}_diff_{}_data.pt'.format(self.train_freq,
                                                      self.differential),
                    'val_{}_diff_{}_data.pt'.format(self.val_freq,
                                                    self.differential),
                    'test_{}_diff_{}_data.pt'.format(self.test_freq,
                                                     self.differential)]
        else:
            raise ValueError('Split mode \"{}\" not supported!'.format(self.split_mode))

    @property
    def processed_file_names(self):
        if self.split_mode == 'file_ids' or self.differential:
            return ['train_{}_diff_{}_data.pt'.format(self.train_sim_ids,
                                                      self.differential),
                    'val_{}_diff_{}_data.pt'.format(self.val_sim_ids,
                                                    self.differential),
                    'test_{}_diff_{}_data.pt'.format(self.test_sim_ids,
                                                     self.differential)]
        elif self.split_mode == 'percentage':
            return ['train_{}_diff_{}_data.pt'.format(self.train_freq,
                                                      self.differential),
                    'val_{}_diff_{}_data.pt'.format(self.val_freq,
                                                    self.differential),
                    'test_{}_diff_{}_data.pt'.format(self.test_freq,
                                                     self.differential)]
        else:
            raise ValueError('Split mode \"{}\" not supported!'.format(self.split_mode))

    def download(self, force=False):
        # Download dataset only if the download folder is not found
        # print('self.download_dir: {}'.format(self.download_dir))
        # if not os.path.exists(self.download_dir) or force:
        #     raise NotImplementedError('Not yet moved dataset to shared drive folder')
        #     # Download tfrecord file if not found...
        #     print('Downloading dataset file, this will take a while...')
        #     radar_online = '1uJ9HTlduxTfSnz91-n8_8-fleUgkUPWB'
        #     tmp_download_folder = os.path.join(os.getcwd(), 'dwn')
        #     if not os.path.exists(tmp_download_folder):
        #         os.makedirs(tmp_download_folder)
        #     download_file_from_google_drive(radar_online, os.path.join(tmp_download_folder, 'RADAR.zip'))
        #     # Extract zip files from downloaded dataset
        #     zf = zipfile.ZipFile(os.path.join(tmp_download_folder, 'RADAR.zip'), 'r')
        #     print('Unzipping dataset...')
        #     zf.extractall(tmp_download_folder)
        #     # Make order into the project folder moving extracted dataset into home and removing temporary download folder
        #     print('Moving dataset to clean repo...')
        #     shutil.move(os.path.join(tmp_download_folder, 'RADAR'), os.path.join(os.getcwd(), self.download_folder))
        #     shutil.rmtree(tmp_download_folder)
        #     # Run _preprocess to activate the extractor which converts the dataset files into tg_graphs
        #     self.convert_dataset_to_tg_graphs()
        return

    def read_files(self):
        # Import stored dictionary of data
        if self.scenario == 'all':
            dwn_dir = os.path.join(self.download_folder, 'IFA_4_existing', '{}_topology'.format(self.topology) \
                if self.topology != 'dfn' else '{}_topology'.format(self.topology.upper()))
            file_names = glob.glob(os.path.join(dwn_dir, '*', '*', '*', '*.txt'))
            dwn_dir = os.path.join(self.download_folder, 'normal', '{}_topology'.format(self.topology) \
                if self.topology != 'dfn' else '{}_topology'.format(self.topology.upper()))
            file_names += glob.glob(os.path.join(dwn_dir, '*.txt'))
        elif self.scenario != 'normal':
            # print('self.download_dir: {}'.format(self.download_dir))
            file_names = glob.glob(os.path.join(self.download_dir, '*', '*', '*', '*.txt'))
        else:
            file_names = glob.glob(os.path.join(self.download_dir, '*.txt'))
        return file_names

    def convert_dataset_to_tg_graphs(self):
        file_names = self.read_files()
        # print('file names: {}'.format(file_names))
        # Rename topology files if they exist
        Extractor.rename_topology_files(file_names)
        file_names = self.read_files()
        # Refactor topology and pit files of they exist
        Extractor.reformat_files(file_names)
        file_names = self.read_files()
        # print('file_names: {}'.format(file_names))
        Extractor(data_dir=self.download_folder,
                  scenario=self.scenario,
                  topology=self.topology,
                  n_attackers=self.n_attackers,
                  train_sim_ids=self.train_sim_ids if self.split_mode == 'file_ids' or self.differential else self.train_freq,
                  val_sim_ids=self.val_sim_ids if self.split_mode == 'file_ids' or self.differential else self.val_freq,
                  test_sim_ids=self.test_sim_ids if self.split_mode == 'file_ids' or self.differential else self.test_freq,
                  simulation_time=self.simulation_time,
                  time_att_start=self.time_att_start,
                  differential=self.differential).run(downloaded_data_file=file_names,
                                                      raw_dir=self.raw_dir,
                                                      raw_file_names=self.raw_file_names,
                                                      split_mode=self.split_mode)

    # def process(self):
    #     print('Start processing')
    #     # Check if it is possible to load the tg raw file
    #     try:
    #         print('self.raw_file_names[0]: {}'.format(self.raw_file_names[0]))
    #         data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[0]))
    #         print('Try worked')
    #     except FileNotFoundError:
    #         print('Inside exception')
    #         # Check if the dataset is already downloaded or not
    #         # Gather real names of files
    #         existing_file_names = glob.glob(os.path.join(self.download_dir, '*', '*.txt'))
    #         # If the two sets don't match it means that the dataset was not downloaded yet
    #         required_file_names = [os.path.join(self.download_dir, name) for name in self.download_file_names]
    #         print('existing_file_names: {}'.format(existing_file_names))
    #         print('required_file_names: {}'.format(required_file_names))
    #         if set(required_file_names) != set(existing_file_names):
    #             print('Didn\'t find the dataset. Downloading it...')
    #             self.download()
    #             print('Running the extractor...')
    #             self.convert_dataset_to_tg_graphs()
    #         else:
    #             print('Tg raw data not found, but dataset was found. Running the extractor...')
    #             self.convert_dataset_to_tg_graphs()
    #     # Iterate over the splits, load raw data, filter and transform them
    #     for index in range(len(self.raw_file_names)):
    #         # Load the raw tg_data
    #         data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[index]))
    #         # Apply pre_filter and pre_transform if necessary
    #         if self.pre_filter is not None:
    #             data_list = [data for data in data_list if self.pre_filter(data)]
    #         if self.pre_transform is not None:
    #             data_list = [self.pre_transform(data) for data in data_list]
    #         # Store
    #         self.store_processed_data(data_list, self.processed_paths[index])

    def process(self):
        print('Start processing')
        # Iterate over the splits, load raw data, filter and transform them
        for index in range(len(self.raw_file_names)):
            # Check if it is possible to load the tg raw file
            try:
                # print('self.raw_file_names[index]: {}'.format(self.raw_file_names[index]))
                data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[index]))
                print('PyTorch geometric raw files found!')
                # Apply pre_filter and pre_transform if necessary
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                # Store
                self.store_processed_data(data_list, self.processed_paths[index])
            except (FileNotFoundError, RuntimeError):
                print('PyTorch geometric raw files not found!')
                # Check if the dataset is already downloaded or not
                # Gather real names of files
                existing_file_names = glob.glob(os.path.join(self.download_dir, '*', '*.txt'))
                # If the two sets don't match it means that the dataset was not downloaded yet
                required_file_names = [os.path.join(self.download_dir, name) for name in self.download_file_names]
                # print('existing_file_names: {}'.format(existing_file_names))
                # print('required_file_names: {}'.format(required_file_names))
                if set(required_file_names) != set(existing_file_names):
                    print('Didn\'t find the dataset. Downloading it...')
                    self.download()
                    print('Running the extractor...')
                    self.convert_dataset_to_tg_graphs()
                else:
                    print('Tg raw data not found, but dataset was found. Running the extractor...')
                    self.convert_dataset_to_tg_graphs()
                # Load the raw tg_data
                data_list = torch.load(os.path.join(self.raw_dir, self.raw_file_names[index]))
                # Apply pre_filter and pre_transform if necessary
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                # Store
                self.store_processed_data(data_list, self.processed_paths[index])

    def store_processed_data(self, data_list, name):
        print('Storing data files to {}...'.format(name))
        # print('data_list: {}'.format(data_list))
        if data_list == []:
            data_list = [Data()]
        data, slices = self.collate(data_list)
        # Create tg processed folder if it doesn't exist
        pr_path = os.path.join(*name.split('/')[:-1])
        # print('pr_path: {}'.format(pr_path))
        if not os.path.exists(pr_path):
            os.makedirs(pr_path)
        torch.save((data, slices), name)

    def get_freq_data(self, frequencies=[8, 16, 32, 64]):
        # Return all graphs of simulations having specified attack frequency
        # Filter graphs by train_scenario and get only those graphs that have attack_is_on==True
        data = [data for data in self if data['frequency'] in frequencies]
        # print('Number of filtered graphs over {} frequencies: {}'.format(frequencies,
        #                                                                  len(data)))
        return data

    def get_all_attack_data(self, frequencies=None):
        # Return all graphs where a specific attack is active
        # Filter graphs by train_scenario and get only those graphs that have attack_is_on==True
        datas = self.get_all_data(frequencies=frequencies)
        # print('Number of non-filtered graphs: {}'.format(len(datas)))
        data = [data for data in datas if data['attack_is_on'] == 1]
        # print('Number of filtered graphs over active attack: {}'.format(len(data)))
        return data

    def get_all_legitimate_data(self, frequencies=None):
        # Return all graphs where no attack is active
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        datas = self.get_all_data(frequencies=frequencies)
        # print('Number of non-filtered graphs: {}'.format(len(datas)))
        # for data in datas:
        #     print('data: {}'.format(data))
        #     break
        data = [data for data in datas if data['attack_is_on'] == 0]
        # print('Number of filtered graphs: {}'.format(len(data)))
        return data

    def get_only_normal_data(self, frequencies=None):
        # Return all graphs where no attack is active
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        datas = self.get_all_data(frequencies=frequencies)
        # print('Number of non-filtered graphs: {}'.format(len(datas)))
        # counter = 0
        # for data in datas:
        #     print('data[train_scenario]: {}'.format(data['train_scenario']))
        #     counter += 1
        #     if counter >= 10:
        #         break
        data = [data for data in datas if data['train_scenario'] == 0]
        # print('Number of filtered graphs: {}'.format(len(data)))
        return data

    def get_randomly_sampled_legitimate_data(self, frequencies=None, p=0.3):
        normal_data = self.get_only_normal_data(frequencies=frequencies)
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        all_legitime_data = self.get_all_legitimate_data(frequencies=frequencies)
        data = normal_data + random.sample(all_legitime_data, math.floor(p * len(all_legitime_data)))
        # print('Number of filtered graphs: {}'.format(len(data)))
        return data

    def get_sampled_legitimate_data(self, n_attackers=11, frequencies=None):
        normal_data = self.get_only_normal_data(frequencies=frequencies)
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        all_legitimate_data = self.get_all_legitimate_data(frequencies=frequencies)
        att_minus_one_data = [data for data in all_legitimate_data if data.n_attackers == n_attackers]
        data = normal_data + att_minus_one_data
        # print('Number of filtered graphs: {}'.format(len(data)))
        return data

    def get_all_data(self, frequencies=None):
        # Return all graphs
        # Gather all graphs over all scenarios
        # If frequencies are specified then filter data by frequencies
        if frequencies:
            data = self.get_freq_data(frequencies=frequencies)
        else:
            data = [data for data in self]
        return data

    def get_data_dict(self, frequencies=None):
        # Return all graphs
        # Gather all graphs over all scenarios
        # If frequencies are specified then filter data by frequencies
        if frequencies:
            data = self.get_freq_data(frequencies=frequencies)
        else:
            data = [data for data in self]
        # print('\n\n\nlen(data): {}\n\n\n'.format(len(data)))
        # Define empty dictionary
        data_dictionary = {0: []}
        sim_index = 0
        last_time = 1
        for sample in data:
            # print('sim_index: {}'.format(sim_index))
            # print('last_time: {}'.format(last_time))
            # print('sample.time.numpy().item(): {}'.format(sample.time.numpy().item()))
            if sample.time.numpy().item() != last_time + 1:
                sim_index += 1
                last_time = 2
                data_dictionary[sim_index] = []
            else:
                last_time += 1
            data_dictionary[sim_index].append(sample)
        # refine dictionary to drop empty samples
        counter = 0
        while counter < len(list(data_dictionary.keys())):
            index = list(data_dictionary.keys())[counter]
            if data_dictionary[index] == []:
                del data_dictionary[index]
            else:
                counter += 1
        # print('data_dictionary: {}'.format(data_dictionary))
        return data_dictionary

    def count_benign_data(self):
        return len([1 for data in self if data['attack_is_on'] == 0])

    def count_malicious_data(self):
        return len([1 for data in self if data['attack_is_on'] == 1])
