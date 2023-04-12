# from gnn4ifa.data import IfaDataset
#
# dataset = IfaDataset(root='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
#                      download_folder='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data')
# print('dataset.raw_dir: {}'.format(dataset.raw_dir))
# print('dataset.processed_dir: {}'.format(dataset.processed_dir))

import argparse
# Import my modules
from baselines.bin import Tester


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for training of GNN.')
    parser.add_argument('--download_folder', default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data',
                        help='folder where raw dataset will be downloaded')
    parser.add_argument('--dataset_folder',
                        default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_base',
                        help='folder where tg dataset will be stored')
    parser.add_argument('--train_scenario', default='existing',
                        help='simulations train_scenario to consider')
    parser.add_argument('--train_topology', default='small',
                        help='simulations train_topology to be used')
    parser.add_argument('--test_scenario', default='existing',
                        help='simulations train_scenario to consider')
    parser.add_argument('--test_topology', default='small',
                        help='simulations train_topology to be used')
    parser.add_argument('--frequencies', nargs="+", type=int, default=None,  # default=[8,12,32,64]
                        help='Frequencies of attacks in the simulations to be used as training')
    parser.add_argument('--train_sims', nargs="+", type=int, default=[1],  # [1, 2, 3],
                        help='IDs list of simulations to be used as training')
    parser.add_argument('--val_sims', nargs="+", type=int, default=[2, 3],  # [4],
                        help='IDs list of simulations to be used as validation')
    parser.add_argument('--test_sims', nargs="+", type=int, default=[4, 5],  # [5],
                        help='IDs list of simulations to be used as test')
    parser.add_argument('--sim_time', type=int, default=300,
                        help='Time of simulations')
    parser.add_argument('--time_att_start', type=int, default=50,
                        help='Attack start time in simulations')
    parser.add_argument('--model', default='svm',
                        help='model to be used for training')
    parser.add_argument('--data_mode', default='avg',
                        help='mode to be used for formatting data')
    parser.add_argument('--feat_set', nargs="+", type=str, default='all',
                        help='list of features to be used in formatting the dataset')
    parser.add_argument('--percentile', default=0.99, type=float,
                        help='percentile to be used during anomaly detection')
    parser.add_argument('--out_folder', default='outputs_base',
                        help='folder where outputs are stored (best trained models, etc.)')
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    tester = Tester(dataset_folder=args.dataset_folder,
                    download_dataset_folder=args.download_folder,
                    train_scenario=args.train_scenario,
                    train_topology=args.train_topology,
                    test_scenario=args.test_scenario,
                    test_topology=args.test_topology,
                    frequencies=args.frequencies,
                    train_sim_ids=args.train_sims,
                    val_sim_ids=args.val_sims,
                    test_sim_ids=args.test_sims,
                    simulation_time=args.sim_time,
                    time_att_start=args.time_att_start,
                    chosen_model=args.model,
                    data_mode=args.data_mode,
                    feat_set=args.feat_set,
                    out_path=args.out_folder)
    tester.run()


if __name__ == '__main__':
    args = gather_settings()
    main(args)
