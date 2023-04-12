import math
import os
import random
import glob
import pickle
# Import modules
from .extractor import Extractor


class IfaDatasetNonML:
    def __init__(self,
                 root='ifa_data_non_ml_baselines',
                 download_folder='ifa_data',
                 scenario='existing',
                 topology='small',
                 n_attackers=None,
                 simulation_time=300,
                 time_att_start=50):
        print(f'Setup:\n'
              f'root={root}\n'
              f'download_folder={download_folder}\n'
              f'scenario={scenario}\n'
              f'topology={topology}\n'
              f'n_attackers={n_attackers}\n'
              f'simulation_time={simulation_time}\n'
              f'time_att_start={time_att_start}\n')
        self.download_folder = download_folder
        assert scenario in ['existing', 'non_existing', 'normal', 'all']
        self.scenario = scenario
        assert topology in ['small', 'dfn', 'large']
        self.topology = topology
        self.n_attackers = n_attackers
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        self.root = root
        self.samples = None
        self.process()

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
        file_names = ['filefilefile']
        return file_names

    @property
    def sample_file_name(self):
        return 'data.pkl'

    def download(self, force=False):
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

    def convert_dataset_to_samples(self):
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
                  simulation_time=self.simulation_time,
                  time_att_start=self.time_att_start).run(downloaded_data_file=file_names,
                                                          raw_dir=self.raw_dir,
                                                          sample_file_name=self.sample_file_name)

    def load_samples(self):
        with open(os.path.join(self.raw_dir, self.sample_file_name), 'rb') as file:
            samples = pickle.load(file)
        return samples

    def process(self):
        print('Start processing')
        # Check if it is possible to load the tg raw file
        try:
            # print('self.raw_file_names[index]: {}'.format(self.raw_file_names[index]))
            self.samples = self.load_samples()
            print('Samples files found! We will use them!')
        except FileNotFoundError:
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
                self.convert_dataset_to_samples()
            else:
                print('Tg raw data not found, but dataset was found. Running the extractor...')
                self.convert_dataset_to_samples()
            # Load the samples
            self.samples = self.load_samples()

    def get_samples(self):
        assert self.samples is not None
        return self.samples

    def get_freq_data(self, frequencies=[8, 16, 32, 64]):
        assert self.samples is not None
        # Return all graphs of simulations having specified attack frequency
        # Filter graphs by train_scenario and get only those graphs that have attack_is_on==True
        data = [data for data in self.samples if data.get_frequency() in frequencies]
        return data

    def get_all_attack_data(self, frequencies=None):
        # Return all graphs where a specific attack is active
        # Filter graphs by train_scenario and get only those graphs that have attack_is_on==True
        datas = self.get_all_data(frequencies=frequencies)
        data = [data for data in datas if data.get_label() == 1]
        return data

    def get_all_legitimate_data(self, frequencies=None):
        # Return all graphs where no attack is active
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        datas = self.get_all_data(frequencies=frequencies)
        data = [data for data in datas if data.get_label() == 0]
        return data

    def get_only_normal_data(self, frequencies=None):
        # Return all graphs where no attack is active
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        datas = self.get_all_data(frequencies=frequencies)
        data = [data for data in datas if data.get_train_scenario() == 0]
        return data

    def get_randomly_sampled_legitimate_data(self, frequencies=None, p=0.3):
        normal_data = self.get_only_normal_data(frequencies=frequencies)
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        all_legitime_data = self.get_all_legitimate_data(frequencies=frequencies)
        data = normal_data + random.sample(all_legitime_data, math.floor(p * len(all_legitime_data)))
        return data

    def get_sampled_legitimate_data(self, n_attackers=11, frequencies=None):
        normal_data = self.get_only_normal_data(frequencies=frequencies)
        # Gather all graphs over all scenarios and get only those graphs that have attack_is_on==False
        all_legitimate_data = self.get_all_legitimate_data(frequencies=frequencies)
        att_minus_one_data = [data for data in all_legitimate_data if data.n_attackers == n_attackers]
        data = normal_data + att_minus_one_data
        return data

    def get_all_data(self, frequencies=None):
        # Return all graphs
        # Gather all graphs over all scenarios
        # If frequencies are specified then filter data by frequencies
        if frequencies:
            data = self.get_freq_data(frequencies=frequencies)
        else:
            data = [data for data in self.samples]
        return data

    def get_data_dict(self, frequencies=None):
        # Return all graphs
        # Gather all graphs over all scenarios
        # If frequencies are specified then filter data by frequencies
        if frequencies:
            data = self.get_freq_data(frequencies=frequencies)
        else:
            data = [data for data in self.samples]
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
        # print('data_dictionary: {}'.format(data_dictionary))
        return data_dictionary

    def count_benign_data(self):
        return len([1 for data in self.samples if data.get_label() == 0])

    def count_malicious_data(self):
        return len([1 for data in self.samples if data.get_label() == 1])
