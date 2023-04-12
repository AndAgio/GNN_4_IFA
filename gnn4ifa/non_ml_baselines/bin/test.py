import os
# Import my modules
from non_ml_baselines.data import IfaDatasetNonML
from non_ml_baselines.models import Poseidon, CooperativeFilter, CongestionAware, Coordination, Chokifa
from non_ml_baselines.metrics import Accuracy, F1Score, Tpr, Fpr, Precision
from non_ml_baselines.utils import get_data_loader, get_labels_scenario_dict, get_labels_attacker_type_dict, \
    get_labels_topology_dict


class Tester:
    def __init__(self,
                 dataset_folder='ifa_data_non_ml_baselines',
                 download_dataset_folder='ifa_data',
                 test_scenario='existing',
                 test_topology='small',
                 frequencies=None,
                 attackers='fixed',
                 n_attackers=None,
                 simulation_time=300,
                 time_att_start=50,
                 chosen_detector='poseidon',
                 out_path='outputs'):
        # Dataset related variables
        self.dataset_folder = dataset_folder
        self.download_dataset_folder = download_dataset_folder
        self.test_scenario = test_scenario
        self.test_topology = test_topology
        self.frequencies = frequencies
        self.attackers = attackers
        self.n_attackers = n_attackers
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        # Model related features
        self.chosen_metrics = ['acc', 'f1', 'tpr', 'fpr', 'precision']
        assert chosen_detector in ['poseidon', 'cooperative_filter', 'congestion_aware', 'coordination', 'chokifa']
        self.chosen_detector = chosen_detector
        # Training related variables
        self.out_path = os.path.join(os.getcwd(), out_path)
        # Get dataset and set up the trainer
        self.detector = None
        self.load_detector()
        self.dataset = None
        self.get_dataset()
        self.metrics = None
        self.setup_metrics()

    def get_dataset(self):
        # Get dataset
        print('Extracting dataset. This may take a while...')
        self.dataset = IfaDatasetNonML(root=self.dataset_folder,
                                       download_folder=self.download_dataset_folder,
                                       scenario=self.test_scenario,
                                       topology=self.test_topology,
                                       simulation_time=self.simulation_time,
                                       time_att_start=self.time_att_start)
        self.dataset = self.dataset.get_samples()
        print('Number of test examples: {}'.format(len(self.dataset)))

    def setup_metrics(self):
        # Setup metrics depending on the choice
        self.metrics = {}
        for metric in self.chosen_metrics:
            if metric == 'acc':
                self.metrics[metric] = Accuracy()
            elif metric == 'f1':
                self.metrics[metric] = F1Score()
            elif metric == 'tpr':
                self.metrics[metric] = Tpr()
            elif metric == 'fpr':
                self.metrics[metric] = Fpr()
            elif metric == 'precision':
                self.metrics[metric] = Precision()
            else:
                raise ValueError('The metric {} is not available in classification mode!'.format(metric))

    def load_detector(self):
        self.detector = {'poseidon': Poseidon(),
                         'cooperative_filter': CooperativeFilter(),
                         'congestion_aware': CongestionAware(),
                         'coordination': Coordination(),
                         'chokifa': Chokifa(),
                         }[self.chosen_detector]

    def run(self):
        print('Testing model {} on topology {}...'.format(self.chosen_detector, self.test_topology))
        self.detector.test(dataset=self.dataset,
                           metrics=self.metrics,
                           verbose=True)
