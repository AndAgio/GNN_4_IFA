import os
import time
import psutil
import random
from memory_profiler import profile
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
# Import my modules
from gnn4ifa.data import IfaDataset
from gnn4ifa.transforms import NormalizeFeatures, RandomNodeMasking
from gnn4ifa.models import Classifier, AutoEncoder
from gnn4ifa.metrics import Accuracy, F1Score, MSE, MAE
from gnn4ifa.utils import get_data_loader, get_labels_scenario_dict, get_labels_attacker_type_dict, \
    get_labels_topology_dict, timeit


class Tester():
    def __init__(self,
                 dataset_folder='ifa_data_tg',
                 download_dataset_folder='ifa_data',
                 train_scenario='existing',
                 train_topology='small',
                 test_scenario='existing',
                 test_topology='small',
                 frequencies=None,
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
                 chosen_model='class_gcn_2x100_mean',
                 masking=False,
                 percentile=0.99,
                 out_path='outputs'):
        # Dataset related variables
        self.dataset_folder = dataset_folder
        self.download_dataset_folder = download_dataset_folder
        self.train_scenario = train_scenario
        self.train_topology = train_topology
        self.test_scenario = test_scenario
        self.test_topology = test_topology
        self.frequencies = frequencies
        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids
        self.train_freq = train_freq
        self.val_freq = val_freq
        self.test_freq = test_freq
        self.split_mode = split_mode
        self.simulation_time = simulation_time
        self.time_att_start = time_att_start
        print(f'differential={differential}')
        if differential and chosen_model.split('_')[0] == 'class':
            raise ValueError('Differential is not available for classification model')
        self.differential = differential
        # Model related variables
        if chosen_model.split('_')[0] == 'class':
            self.mode = 'class'
            # Metrics related features
            self.chosen_metrics = ['acc', 'f1']
            self.metric_to_check = 'acc'
            self.lr_metric_to_check = 'acc'
        elif chosen_model.split('_')[0] == 'anomaly':
            self.mode = 'anomaly'
            # Metrics related features
            self.chosen_metrics = ['mse', 'mae']
            self.metric_to_check = 'mse'
            self.lr_metric_to_check = 'mse'
            self.masking = masking
            self.percentile = percentile
        else:
            raise ValueError('Model should indicate training mode between classification and anomaly!')
        self.chosen_model = chosen_model
        # Training related variables
        self.out_path = os.path.join(os.getcwd(), out_path)
        self.trained_models_folder = os.path.join(self.out_path, 'trained_models')
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Get dataset and setup the trainer
        self.get_dataset()
        self.get_model()

    @profile
    def get_dataset(self):
        # Get dataset
        if self.mode == 'class':
            transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x'])])
        elif self.mode == 'anomaly':
            if self.masking:
                transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x']),
                                                   RandomNodeMasking(sampling_strategy='local',
                                                                     sampling_probability=0.3)
                                                   ])
            else:
                transform = tg.transforms.Compose([NormalizeFeatures(attrs=['x'])])
        else:
            raise ValueError('Something wrong with selected mode between classification and anomaly!')
        # Training
        self.train_dataset = IfaDataset(root=self.dataset_folder,
                                        download_folder=self.download_dataset_folder,
                                        transform=transform,
                                        scenario=self.train_scenario if self.mode == 'class' else 'all' if self.differential else 'normal',
                                        topology=self.train_topology,
                                        train_sim_ids=self.train_sim_ids,
                                        val_sim_ids=self.val_sim_ids,
                                        test_sim_ids=self.test_sim_ids,
                                        train_freq=self.train_freq,
                                        val_freq=self.val_freq,
                                        test_freq=self.test_freq,
                                        split_mode=self.split_mode,
                                        simulation_time=self.simulation_time,
                                        time_att_start=self.time_att_start,
                                        differential=self.differential,
                                        split='train')
        # print('Number of training examples: {}'.format(len(self.train_dataset)))
        # Test
        self.test_dataset = IfaDataset(root=self.dataset_folder,
                                       download_folder=self.download_dataset_folder,
                                       transform=transform,
                                       # scenario='all' if self.mode == 'class' else self.train_scenario,
                                       scenario=self.test_scenario,  # self.train_scenario,
                                       topology=self.test_topology,
                                       train_sim_ids=self.train_sim_ids if self.mode == 'class' else [0],
                                       val_sim_ids=self.val_sim_ids if self.mode == 'class' else [0],
                                       test_sim_ids=self.test_sim_ids if self.mode == 'class' else [1, 2, 3, 4, 5],
                                       train_freq=self.train_freq,
                                       val_freq=self.val_freq,
                                       test_freq=self.test_freq,
                                       split_mode=self.split_mode,
                                       simulation_time=self.simulation_time,
                                       time_att_start=self.time_att_start,
                                       differential=self.differential,
                                       split='test')
        print('Number of test examples: {}'.format(len(self.test_dataset)))
        if self.mode == 'class':
            self.test_loader = get_data_loader(self.test_dataset,
                                               batch_size=1,
                                               shuffle=False)
        elif self.differential:
            self.test_loader = self.test_dataset.get_data_dict(frequencies=self.frequencies)
            # print('self.test_loader: {}'.format(self.test_loader))
        elif not self.differential:
            self.test_loader = get_data_loader(self.test_dataset.get_all_data(frequencies=self.frequencies),
                                               batch_size=1,
                                               shuffle=False)
        else:
            raise ValueError('Something wrong with test set loading!')
        # Get the number of node features
        self.num_node_features = self.train_dataset.num_features
        print('self.num_node_features:', self.num_node_features)

    def get_model(self):
        # Get the model depending on the string passed by user
        self.load_best_model()
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)
        # Setup metrics depending on the choice
        self.metrics = {'acc': Accuracy(),
                        'f1': F1Score()}

    @profile
    def load_best_model(self):
        if self.mode == 'class':
            if self.split_mode == 'percentage':
                model_name = '{} sce_{} topo_{} diff_{} perc_{} best.pt'.format(self.chosen_model,
                                                                                self.train_scenario,
                                                                                self.train_topology,
                                                                                self.differential,
                                                                                self.train_freq)
            elif self.split_mode == 'file_ids':
                model_name = '{} sce_{} topo_{} diff_{} ids_{} best.pt'.format(self.chosen_model,
                                                                               self.train_scenario,
                                                                               self.train_topology,
                                                                               self.differential,
                                                                               self.train_sim_ids)
            else:
                raise ValueError('Split mode \"{}\" not supported!'.format(self.split_mode))
        elif self.mode == 'anomaly':
            if self.split_mode == 'percentage':
                model_name = '{} mask_{} sce_{} topo_{} diff_{} perc_{} best.pt'.format(self.chosen_model,
                                                                                        self.masking,
                                                                                        self.train_scenario,
                                                                                        self.train_topology,
                                                                                        self.differential,
                                                                                        self.train_freq)
            elif self.split_mode == 'file_ids':
                model_name = '{} mask_{} sce_{} topo_{} diff_{} ids_{} best.pt'.format(self.chosen_model,
                                                                                       self.masking,
                                                                                       self.train_scenario,
                                                                                       self.train_topology,
                                                                                       self.differential,
                                                                                       self.train_sim_ids)
            else:
                raise ValueError('Split mode \"{}\" not supported!'.format(self.split_mode))
        else:
            raise ValueError('Something went wrong with mode selection')
        model_path = os.path.join(self.trained_models_folder, model_name)
        self.model = torch.load(model_path)
        # Move model to GPU or CPU
        self.model = self.model.to(self.device)

    @profile
    def run(self):
        start = time.time()
        print('Start testing...')
        self.test()
        stop = time.time()
        print('The average CPU usage during training is {} %'.format(psutil.cpu_percent(stop - start)))

    # @torch.no_grad()
    # def test(self):
    #     # Set the valuer to be non trainable
    #     self.model.eval()
    #     if self.mode == 'class':
    #         self._test_class()
    #     elif self.mode == 'anomaly':
    #         self._test_anomaly()
    #     else:
    #         raise ValueError('Something wrong with mode selection')
    #
    # @torch.no_grad()
    # def _test_class(self):
    #     all_preds = None
    #     all_labels = None
    #     for batch_index, data in enumerate(self.test_loader):
    #         batch_preds, batch_labels = self.test_step(data)
    #         # Append batch predictions and labels to the list containing every prediction and every label
    #         if batch_index == 0:
    #             all_preds = batch_preds
    #             all_labels = batch_labels
    #         else:
    #             all_preds = torch.cat((all_preds, batch_preds), dim=0)
    #             all_labels = torch.cat((all_labels, batch_labels), dim=0)
    #         # Compute metrics over predictions
    #         scores = {}
    #         for metric_name, metric_object in self.metrics.items():
    #             scores[metric_name] = metric_object.compute(y_pred=all_preds,
    #                                                         y_true=all_labels)
    #         self.print_test_message(index_batch=batch_index, metrics=scores)
    #     print()
    #
    # @torch.no_grad()
    # def _test_anomaly(self):
    #     threshold_metric = 'mse'
    #     # Get threshold from training
    #     threshold = self.get_threshold(metric=threshold_metric)
    #     all_preds = None
    #     all_labels = None
    #     for batch_index, data in enumerate(self.test_loader):
    #         batch_preds, batch_labels = self.test_step(data, threshold, threshold_metric=threshold_metric)
    #         # Append batch predictions and labels to the list containing every prediction and every label
    #         if batch_index == 0:
    #             all_preds = batch_preds
    #             all_labels = batch_labels
    #         else:
    #             all_preds = torch.cat((all_preds, batch_preds), dim=0)
    #             all_labels = torch.cat((all_labels, batch_labels), dim=0)
    #         # Compute metrics over predictions
    #         scores = {}
    #         for metric_name, metric_object in self.metrics.items():
    #             scores[metric_name] = metric_object.compute(y_pred=all_preds,
    #                                                         y_true=all_labels)
    #         self.print_test_message(index_batch=batch_index, metrics=scores)
    #     print()
    #
    # @torch.no_grad()
    # def get_threshold(self, metric='mae'):
    #     print('Computing {}s over legitimate training samples'.format(metric.upper()))
    #     values = []
    #     loader = get_data_loader(self.train_dataset.get_all_legitimate_data(),
    #                              batch_size=1,
    #                              shuffle=True)
    #     for batch_index, data in enumerate(loader):
    #         data = data.to(self.device)
    #         # Pass graphs through model
    #         y_pred = self.model(data)
    #
    #         if not self.masking:
    #             # Get labels from data
    #             y_true = data.x.float()
    #         else:
    #             # Get masks for masked nodes and edges to be used to optimize model
    #             nodes_mask = data.masked_nodes_indices
    #             # Get labels for node and edge predictions
    #             y_true = data.original_x[nodes_mask].float()
    #             # Mask the predictions over the masked nodes only
    #             y_pred = y_pred[nodes_mask]
    #
    #         if metric == 'mae':
    #             values_samples = MAE.compute(y_pred=y_pred, y_true=y_true).item()
    #         elif metric == 'mse':
    #             values_samples = MSE.compute(y_pred=y_pred, y_true=y_true).item()
    #         else:
    #             raise ValueError('Metric {} is not available for computing threshold')
    #         # print('values_samples: {}'.format(values_samples))
    #         values.append(values_samples)
    #     # Get threshold value depending on the percentile given
    #     sorted_values = np.sort(values)
    #     threshold_index = int(len(sorted_values) * self.percentile)
    #     threshold = sorted_values[threshold_index]
    #     print('sorted_values: {}'.format(sorted_values))
    #     print('{} threshold obtained: {}'.format(metric.upper(), threshold))
    #     return threshold
    #
    # @torch.no_grad()
    # def test_step(self, data, threshold=None, threshold_metric='mae'):
    #     data = data.to(self.device)
    #     if self.mode == 'anomaly':
    #         # Pass graphs through model
    #         preds = self.model(data)
    #
    #         if not self.masking:
    #             # Get labels from data
    #             y_true = data.x.float()
    #         else:
    #             # Get masks for masked nodes and edges to be used to optimize model
    #             nodes_mask = data.masked_nodes_indices
    #             # Get labels for node and edge predictions
    #             y_true = data.original_x[nodes_mask].float()
    #             # Mask the predictions over the masked nodes only
    #             preds = preds[nodes_mask]
    #
    #         if threshold_metric == 'mae':
    #             preds_mses = MAE.compute(y_pred=preds, y_true=y_true)
    #         elif threshold_metric == 'mse':
    #             preds_mses = MSE.compute(y_pred=preds, y_true=y_true)
    #         else:
    #             raise ValueError('Metric {} is not available for computing threshold')
    #         y_pred = torch.as_tensor(np.array([preds_mses > threshold]).astype('int'))
    #         # print('preds: {} -> original_x: {}'.format(preds,
    #         #                                            y_true))
    #         print('preds_mses: {:.5f} -> label: {} -> '
    #               'prediction: {} -> freq: {}'.format(preds_mses,
    #                                                   data.attack_is_on.item(),
    #                                                   y_pred.item(),
    #                                                   data.frequency.item()))
    #     elif self.mode == 'class':
    #         y_pred = self.model(data)
    #     # Get labels from data
    #     y_true = data.attack_is_on
    #     # Return predictions and labels
    #     return y_pred, y_true
    #
    # def print_test_message(self, index_batch, metrics):
    #     message = '| '
    #     bar_length = 10
    #     total_batches = len(self.test_loader)
    #     progress = float(index_batch) / float(total_batches)
    #     if progress >= 1.:
    #         progress = 1
    #     block = int(round(bar_length * progress))
    #     message += '[{}] | TEST: '.format('=' * block + ' ' * (bar_length - block))
    #     if metrics is not None:
    #         metrics_message = ''
    #         for metric_name, metric_value in metrics.items():
    #             metrics_message += '{}={:.5f} '.format(metric_name,
    #                                                    metric_value)
    #         message += metrics_message
    #     message += '|'
    #     print(message, end='\r')

    @torch.no_grad()
    def test(self):
        # Set the valuer to be non trainable
        self.model.eval()
        if self.mode == 'class':
            self._test_class()
        elif self.mode == 'anomaly' and self.differential:
            self._test_anomaly_differential()
        elif self.mode == 'anomaly' and not self.differential:
            self._test_anomaly()
        else:
            raise ValueError('Something wrong with mode selection')

    @torch.no_grad()
    def _test_class(self):
        all_preds = None
        all_labels = None
        inference_times = []
        metrics_dict = self.get_empty_metrics_dict(mode='sad')
        for batch_index, data in enumerate(self.test_loader):
            start_time = time.time()
            batch_preds, batch_labels = self.test_step(data, metrics=self.metrics)
            inference_times.append((time.time() - start_time) / data.attack_is_on.shape[0])
            # Append batch predictions and labels to the list containing every prediction and every label
            if batch_index == 0:
                all_preds = batch_preds
                all_labels = batch_labels
            else:
                all_preds = torch.cat((all_preds, batch_preds), dim=0)
                all_labels = torch.cat((all_labels, batch_labels), dim=0)
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in self.metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=all_preds,
                                                            y_true=all_labels)
            self.print_test_message(index_batch=batch_index, metrics=scores)
            # Get metrics from single sample, to be added to metrics_dict
            topologies = data.topology.numpy()
            scenarios = data.train_scenario.numpy()
            frequencies = data.frequency.numpy()
            attackers_types = data.attackers_type.numpy()
            n_attackerss = data.n_attackers.numpy()
            for sample_index in range(data.attack_is_on.shape[0]):
                topology = get_labels_topology_dict()[topologies[sample_index]]
                scenario = get_labels_scenario_dict()[scenarios[sample_index]]
                frequency = frequencies[sample_index]
                attackers_type = get_labels_attacker_type_dict()[attackers_types[sample_index]]
                n_attackers = n_attackerss[sample_index]
                # print('Test sample taken from simulation with'
                #       'topology={}, scenario={}, '
                #       'attackers_type={}, frequency={}, '
                #       'n_attackers={}'.format(topology,
                #                               scenario,
                #                               attackers_type,
                #                               frequency,
                #                               n_attackers))
                metrics_dict[topology][scenario][frequency][attackers_type][n_attackers]['preds'].append(
                    batch_preds[sample_index])
                metrics_dict[topology][scenario][frequency][attackers_type][n_attackers]['labels'].append(
                    batch_labels[sample_index])
        print()
        inference_times = [inf * 1000 for inf in inference_times]
        print(u'Inference time: {:.3f} \u00B1 {:.3f} ms over {} samples'.format(np.mean(inference_times),
                                                                                np.std(inference_times),
                                                                                len(inference_times)))
        self.format_and_print_stats(metrics_dict, mode='sad')

    @torch.no_grad()
    def _test_anomaly(self):
        threshold_metric = 'mse'
        # Get threshold from training
        threshold = self.get_threshold(metric=threshold_metric)
        # Define classification metrics
        metrics = {'acc': Accuracy(),
                   'f1': F1Score()}
        all_preds = None
        all_labels = None
        for batch_index, data in enumerate(self.test_loader):
            batch_preds, batch_labels = self.test_step(data, metrics, threshold, threshold_metric=threshold_metric)
            # Append batch predictions and labels to the list containing every prediction and every label
            if batch_index == 0:
                all_preds = batch_preds
                all_labels = batch_labels
            else:
                all_preds = torch.cat((all_preds, batch_preds), dim=0)
                all_labels = torch.cat((all_labels, batch_labels), dim=0)
            # Compute metrics over predictions
            scores = {}
            for metric_name, metric_object in metrics.items():
                scores[metric_name] = metric_object.compute(y_pred=all_preds,
                                                            y_true=all_labels)
            self.print_test_message(index_batch=batch_index, metrics=scores)
        print()

    @torch.no_grad()
    def _test_anomaly_differential(self):
        threshold_metric = 'mse'
        # Get threshold from training
        threshold = self.get_threshold(metric=threshold_metric)
        # Define empty list for simulations metrics
        simulations_false_alarms = []
        simulations_true_alarms = []
        simulations_exact_alarms = []
        inference_times = []
        metrics_dict = self.get_empty_metrics_dict(mode='uad')
        # Iterate over all simulations belonging to the test set
        for sim_index, simulation in self.test_loader.items():
            # print('simulation: {}'.format(simulation))
            # print('simulation[0]: {}'.format(simulation[0]))
            topology = get_labels_topology_dict()[simulation[0].topology.item()]
            scenario = get_labels_scenario_dict()[simulation[0].train_scenario.item()]
            frequency = simulation[0].frequency.item()
            attackers_type = get_labels_attacker_type_dict()[simulation[0].attackers_type.item()]
            n_attackers = simulation[0].n_attackers.item()
            # print('Testing on simulation with topology={}, scenario={}, '
            #       'attackers_type={}, frequency={}, n_attackers={}'.format(topology,
            #                                                                scenario,
            #                                                                attackers_type,
            #                                                                frequency,
            #                                                                n_attackers))
            # Iterate over each sample of the simulation
            predictions = []
            labels = []
            for sample_index, sample in enumerate(simulation):
                sample = sample.to(self.device)
                start_time = time.time()
                prediction, label = self.test_step(sample, metrics=None,
                                                   threshold=threshold, threshold_metric=threshold_metric)
                inference_times.append(time.time() - start_time)
                # Append prediction and label to the list containing every prediction and label of the simulation
                predictions.append(prediction.numpy().item())
                labels.append(label.numpy().item())
            # print('predictions: {}'.format(predictions))
            # print('labels: {}'.format(labels))
            # Get number of false alarms, true alarms and the behaviour at the
            # starting point of the attack (exact alarm) for the simulation
            false_alarms = 0
            true_alarms = 0
            exact_alarm = 0
            for index in range(len(predictions)):
                if predictions[index] == 1 and labels[index] == 1:
                    true_alarms += 1
                elif predictions[index] == 1 and labels[index] == 0:
                    false_alarms += 1
                else:
                    pass
                if labels[index] == 1 and labels[index - 1] == 0:
                    if predictions[index] == 1 and labels[index] == 1:
                        exact_alarm = 1
            # print('False Alarms: {}'.format(false_alarms))
            # print('True Alarms: {}'.format(true_alarms))
            # print('Exact Alarms: {}'.format(exact_alarm))
            metrics_dict[topology][scenario][frequency][attackers_type][n_attackers]['tps'].append(
                1 if true_alarms >= 1 else 0)
            metrics_dict[topology][scenario][frequency][attackers_type][n_attackers]['fps'].append(false_alarms)
            # print('TPR={}'.format(sum(i>=1 for i in true_alarms)/len(true_alarms)))
            # print('FPR={}'.format(sum(false_alarms)/len(false_alarms)))
            # Append the values to the list of values defining performances over simulations
            simulations_false_alarms.append(false_alarms)
            simulations_true_alarms.append(true_alarms)
            simulations_exact_alarms.append(exact_alarm)
        # Print the results
        # print('True alarms over testing simulations: {}'.format(simulations_true_alarms))
        # print('False alarms over testing simulations: {}'.format(simulations_false_alarms))
        # print('Exact alarm over testing simulations: {}'.format(simulations_exact_alarms))
        # print('Overall TPR = {:.3f}%'.format(100 * sum(i >= 1 for i in simulations_true_alarms) /
        #                                      len(simulations_true_alarms)))
        # print('Overall FPR = {:.3f}%'.format(100 * sum(simulations_false_alarms) /
        #                                      (len(simulations_false_alarms) * self.time_att_start)))
        inference_times = [inf * 1000 for inf in inference_times]
        # print(u'Inference time: {:.3f} \u00B1 {:.3f} ms over {} samples'.format(np.mean(inference_times),
        #                                                                         np.std(inference_times),
        #                                                                         len(inference_times)))
        self.format_and_print_stats(metrics_dict, mode='uad')

    def format_and_print_stats(self, metrics_dict, mode='uad'):
        # print('metrics_dict: {}'.format(metrics_dict))
        if mode == 'uad':
            metrics_dict = self.format_metrics_dict(metrics_dict)
        elif mode == 'sad':
            pass
        else:
            raise ValueError('Mode {} is not available!')
        # print('metrics_dict: {}'.format(metrics_dict))
        interesting_setups_metrics_dict = {freq_name: {n_att: None}
                                           for topo_name, topo_dict in metrics_dict.items()
                                           for scenario_name, scenario_dict in topo_dict.items()
                                           for freq_name, freq_dict in scenario_dict.items()
                                           for att_type_name, att_type_dict in freq_dict.items()
                                           for n_att, data in att_type_dict.items()
                                           if (topo_name == self.test_topology
                                               and scenario_name == 'existing'
                                               and att_type_name == 'variable')}
        for topo_name, topo_dict in metrics_dict.items():
            for scenario_name, scenario_dict in topo_dict.items():
                for freq_name, freq_dict in scenario_dict.items():
                    for att_type_name, att_type_dict in freq_dict.items():
                        for n_att, data in att_type_dict.items():
                            if (topo_name == self.test_topology and scenario_name == 'existing'
                                    and att_type_name == 'variable'):
                                interesting_setups_metrics_dict[freq_name][n_att] = data
        # print('interesting_setups_metrics_dict: {}'.format(interesting_setups_metrics_dict))
        # print('interesting_setups_metrics_dict dataframe:\n{}'.format(pd.DataFrame(interesting_setups_metrics_dict)))
        # Print results to text output file
        output_path = os.path.join(self.out_path, 'results', mode.upper(), 'generalisability')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if mode == 'uad':
            output_file = os.path.join(output_path, 'model={}_topos={}2{}_p={}_d={}.txt'.format(self.chosen_model,
                                                                                                self.train_topology,
                                                                                                self.test_topology,
                                                                                                self.percentile,
                                                                                                1 if self.differential else 0))
        elif mode == 'sad':
            output_file = os.path.join(output_path, 'model={}_topos={}2{}.txt'.format(self.chosen_model,
                                                                                      self.train_topology,
                                                                                      self.test_topology))
        # Compute cumulative metrics
        first = {freq_name: {n_att: None} for freq_name, freq_dict in interesting_setups_metrics_dict.items()
                 for n_att, data in freq_dict.items()}
        for freq_name, freq_dict in interesting_setups_metrics_dict.items():
            for n_att, data in freq_dict.items():
                if mode == 'uad':
                    first[freq_name][n_att] = data['tps']
                elif mode == 'sad':
                    predictions = torch.stack(data['preds'])
                    labels = torch.stack(data['labels'])
                    first[freq_name][n_att] = self.metrics['acc'].compute(y_pred=predictions,
                                                                          y_true=labels) * 100
                else:
                    raise ValueError('Mode {} is not available!')
        first = pd.DataFrame(first)
        second = {freq_name: {n_att: None} for freq_name, freq_dict in interesting_setups_metrics_dict.items()
                  for n_att, data in freq_dict.items()}
        for freq_name, freq_dict in interesting_setups_metrics_dict.items():
            for n_att, data in freq_dict.items():
                if mode == 'uad':
                    second[freq_name][n_att] = data['fps']
                elif mode == 'sad':
                    predictions = torch.stack(data['preds'])
                    labels = torch.stack(data['labels'])
                    second[freq_name][n_att] = self.metrics['f1'].compute(y_pred=predictions,
                                                                          y_true=labels) * 100
        second = pd.DataFrame(second)
        first = first.append(first.mean(axis=0).rename('avg'))
        first['avg'] = first.mean(axis=1)
        print('first dataframe:\n{}'.format(first))
        second = second.append(second.mean(axis=0).rename('avg'))
        second['avg'] = second.mean(axis=1)
        print('second dataframe:\n{}'.format(second))

        with open(output_file, 'w') as f:
            f.write('\n\n\nFIRST METRIC DATAFRAME:\n')
            f.write('{}'.format(first))
            f.write('\n\n\nSECOND METRIC DATAFRAME:\n')
            f.write('{}'.format(second))

        # Store first and second to pickle files for plotting
        import pickle
        print('output_file.split(\'.txt\'): {}'.format(output_file.split('.txt')))
        with open(output_file.split('.txt')[0] + '.pkl', 'wb') as handle:
            pickle.dump({'tps' if mode == 'uad' else 'acc': first,
                         'fps' if mode == 'uad' else 'f1': second}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def format_metrics_dict(self, metrics_dict):
        # Iterate over topologies
        for topo_name, topo_dict in metrics_dict.items():
            # Iterate over scenarios
            for scenario_name, scenario_dict in topo_dict.items():
                # Iterate over frequencies
                for freq_name, freq_dict in scenario_dict.items():
                    # Iterate over attacker types
                    for att_type_name, att_type_dict in freq_dict.items():
                        # Iterate over number of attackers
                        for n_att_name, data in att_type_dict.items():
                            # Normalize data
                            data['tps'] = 100 * sum(data['tps']) / len(data['tps'])
                            data['fps'] = 100 * sum(data['fps']) / (self.time_att_start * len(data['fps']))
        return metrics_dict

    @torch.no_grad()
    def get_empty_metrics_dict(self, mode='uad'):
        empty_dict = {}
        if mode == 'uad':
            loader = self.test_loader
        elif mode == 'sad':
            loader = self.test_dataset.get_data_dict(frequencies=self.frequencies)
        else:
            raise ValueError('Mode {} is not valid!')
        for sim_index, simulation in loader.items():
            # print('simulation: {}'.format(simulation))
            if not simulation:
                continue
            topology = get_labels_topology_dict()[simulation[0].topology.item()]
            scenario = get_labels_scenario_dict()[simulation[0].train_scenario.item()]
            freq = simulation[0].frequency.item()
            attack_type = get_labels_attacker_type_dict()[simulation[0].attackers_type.item()]
            n_att = simulation[0].n_attackers.item()
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
                empty_dict[topology][scenario][freq][attack_type]
            except KeyError:
                empty_dict[topology][scenario][freq][attack_type] = {}
            try:
                empty_dict[topology][scenario][freq][attack_type][n_att]
            except KeyError:
                empty_dict[topology][scenario][freq][attack_type][n_att] = {}
            try:
                empty_dict[topology][scenario][freq][attack_type][n_att]['tps' if mode == 'uad' else 'preds']
            except KeyError:
                empty_dict[topology][scenario][freq][attack_type][n_att]['tps' if mode == 'uad' else 'preds'] = []
            try:
                empty_dict[topology][scenario][freq][attack_type][n_att]['fps' if mode == 'uad' else 'labels']
            except KeyError:
                empty_dict[topology][scenario][freq][attack_type][n_att]['fps' if mode == 'uad' else 'labels'] = []
        return empty_dict

    @torch.no_grad()
    def get_threshold(self, metric='mae'):
        print('Computing {}s over legitimate training samples'.format(metric.upper()))
        values = []
        normal_data = self.train_dataset.get_only_normal_data()
        if self.split_mode == 'percentage':
            n_normal_samples = int(len(normal_data) * self.train_freq)
            normal_data = random.sample(normal_data, n_normal_samples)
        loader = get_data_loader(normal_data,
                                 batch_size=1,
                                 shuffle=True)
        for batch_index, data in enumerate(loader):
            data = data.to(self.device)
            # Pass graphs through model
            y_pred = self.model(data)

            if not self.masking:
                # Get labels from data
                y_true = data.x.float()
            else:
                # Get masks for masked nodes and edges to be used to optimize model
                nodes_mask = data.masked_nodes_indices
                # Get labels for node and edge predictions
                y_true = data.original_x[nodes_mask].float()
                # Mask the predictions over the masked nodes only
                y_pred = y_pred[nodes_mask]

            if metric == 'mae':
                values_samples = MAE.compute(y_pred=y_pred, y_true=y_true).item()
            elif metric == 'mse':
                values_samples = MSE.compute(y_pred=y_pred, y_true=y_true).item()
            else:
                raise ValueError('Metric {} is not available for computing threshold')
            # print('values_samples: {}'.format(values_samples))
            values.append(values_samples)
        # Get threshold value depending on the percentile given
        sorted_values = np.sort(values)
        print('len(values) = {}'.format(len(values)))
        if 0 < self.percentile < 1:
            threshold_index = int(len(sorted_values) * self.percentile)
        elif self.percentile == 1:
            threshold_index = int(len(sorted_values)) - 1
        else:
            raise ValueError('Percentile should be between 0 and 1')
        threshold = sorted_values[threshold_index]
        # print('sorted_values: {}'.format(sorted_values))
        print('{} threshold obtained: {}'.format(metric.upper(), threshold))
        return threshold

    @torch.no_grad()
    def test_step(self, data, metrics, threshold=None, threshold_metric='mae'):
        data = data.to(self.device)
        if self.mode == 'anomaly':
            # Pass graphs through model
            preds = self.model(data)

            if not self.masking:
                # Get labels from data
                y_true = data.x.float()
            else:
                # Get masks for masked nodes and edges to be used to optimize model
                nodes_mask = data.masked_nodes_indices
                # Get labels for node and edge predictions
                y_true = data.original_x[nodes_mask].float()
                # Mask the predictions over the masked nodes only
                preds = preds[nodes_mask]

            if threshold_metric == 'mae':
                preds_mses = MAE.compute(y_pred=preds, y_true=y_true)
            elif threshold_metric == 'mse':
                preds_mses = MSE.compute(y_pred=preds, y_true=y_true)
            else:
                raise ValueError('Metric {} is not available for computing threshold')
            y_pred = torch.as_tensor(np.array([preds_mses > threshold]).astype('int'))
            # print('preds: {} -> original_x: {}'.format(preds,
            #                                            y_true))
            # print('preds_mses: {:.5f} -> label: {} -> '
            #       'prediction: {} -> freq: {}'.format(preds_mses,
            #                                           data.attack_is_on.item(),
            #                                           y_pred.item(),
            #                                           data.frequency.item()))
        elif self.mode == 'class':
            y_pred = self.model(data)
        # Get labels from data
        y_true = data.attack_is_on
        # Return predictions and labels
        return y_pred, y_true

    def print_test_message(self, index_batch, metrics):
        message = '| '
        bar_length = 10
        total_batches = len(self.test_loader)
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}] | TEST: '.format('=' * block + ' ' * (bar_length - block))
        if metrics is not None:
            metrics_message = ''
            for metric_name, metric_value in metrics.items():
                metrics_message += '{}={:.5f} '.format(metric_name,
                                                       metric_value)
            message += metrics_message
        message += '|'
        print(message, end='\r')
