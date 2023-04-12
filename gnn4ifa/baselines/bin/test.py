import os
import time
import numpy as np
import pandas as pd
import pickle
# Import my modules
from baselines.data import Datasetter
from baselines.models import Classifier
from baselines.metrics import Accuracy, F1Score, Tpr, Fpr, Precision
from baselines.utils import get_labels_scenario_dict, get_labels_attacker_type_dict, get_labels_topology_dict


class Tester():
    def __init__(self,
                 dataset_folder='ifa_data_baselines',
                 download_dataset_folder='ifa_data',
                 train_scenario='existing',
                 train_topology='small',
                 test_scenario='existing',
                 test_topology='small',
                 frequencies=None,
                 attackers='fixed',
                 n_attackers=None,
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50,
                 chosen_model='svm',
                 data_mode='cat',
                 feat_set='all',
                 out_path='outputs'):
        # Dataset related variables
        self.dataset_folder = dataset_folder
        self.download_dataset_folder = download_dataset_folder
        self.train_scenario = train_scenario
        self.train_topology = train_topology
        self.test_scenario = test_scenario
        self.test_topology = test_topology
        self.frequencies = frequencies
        self.attackers = attackers
        self.n_attackers = n_attackers
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        # Model related features
        self.chosen_metrics = ['acc', 'f1', 'tpr', 'fpr', 'precision']
        self.chosen_model = chosen_model
        self.data_mode = data_mode
        self.feat_set = feat_set
        # Training related variables
        self.out_path = os.path.join(os.getcwd(), out_path)
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Get dataset and set up the trainer
        self.get_dataset()
        self.setup()

    def get_dataset(self):
        # Get dataset
        # Training
        print('Extracting training dataset. This may take a while...')
        self.datasetter = Datasetter(data_dir=self.download_dataset_folder,
                                     # scenario='all',
                                     scenario=self.test_scenario,
                                     topology=self.test_topology,
                                     n_attackers=self.n_attackers,
                                     train_sim_ids=self.train_sim_ids,
                                     val_sim_ids=self.val_sim_ids,
                                     test_sim_ids=self.test_sim_ids,
                                     simulation_time=self.simulation_time,
                                     time_att_start=self.time_att_start,
                                     mode=self.data_mode,
                                     selected_features=self.feat_set,
                                     out_file=os.path.join(self.dataset_folder,
                                                           '{}'.format(self.feat_set)))
        self.datasetter.run()
        self.test_set = self.datasetter.read_split(split='test').sample(frac=1)

    def setup(self):
        # Get the model depending on the string passed by user
        self.load_best_model()
        # Setup metrics depending on the choice
        self.metrics = {}
        for metric in self.chosen_metrics:
            if metric == 'acc':
                self.metrics[metric] = Accuracy(data_mode=self.data_mode)
            elif metric == 'f1':
                self.metrics[metric] = F1Score(data_mode=self.data_mode)
            elif metric == 'tpr':
                self.metrics[metric] = Tpr(data_mode=self.data_mode)
            elif metric == 'fpr':
                self.metrics[metric] = Fpr(data_mode=self.data_mode)
            elif metric == 'precision':
                self.metrics[metric] = Precision(data_mode=self.data_mode)
            else:
                raise ValueError('The metric {} is not available in classification mode!'.format(metric))

    def load_best_model(self):
        model_name = '{} dm_{} fs_{} sce_{} topo_{} ts_{}.pkl'.format(self.chosen_model,
                                                                      self.data_mode,
                                                                      self.feat_set,
                                                                      self.train_scenario,
                                                                      self.train_topology,
                                                                      self.train_sim_ids)
        model_path = os.path.join(self.trained_models_folder, model_name)
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def run(self):
        # Test model
        print('Testing model {} from train topology {} to test topology {}...'.format(self.chosen_model,
                                                                                      self.train_topology,
                                                                                      self.test_topology))
        self.model.test(dataset=self.test_set,
                        metrics=self.metrics,
                        verbose=True)
