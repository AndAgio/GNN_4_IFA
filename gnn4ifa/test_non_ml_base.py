import argparse
# Import my modules
from non_ml_baselines.bin import Tester


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for training of GNN.')
    parser.add_argument('--download_folder', default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data',
                        help='folder where raw dataset will be downloaded')
    parser.add_argument('--dataset_folder',
                        default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_non_ml_baselines',
                        help='folder where tg dataset will be stored')
    parser.add_argument('--test_scenario', default='existing',
                        help='simulations train_scenario to consider')
    parser.add_argument('--test_topology', default='small',
                        help='simulations train_topology to be used')
    parser.add_argument('--frequencies', nargs="+", type=int, default=None,  # default=[8,12,32,64]
                        help='Frequencies of attacks in the simulations to be used as training')
    parser.add_argument('--sim_time', type=int, default=300,
                        help='Time of simulations')
    parser.add_argument('--time_att_start', type=int, default=50,
                        help='Attack start time in simulations')
    parser.add_argument('--model', default='poseidon',
                        help='model to be used for training')
    parser.add_argument('--out_folder', default='outputs_non_ml_base',
                        help='folder where outputs are stored (best trained models, etc.)')
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    tester = Tester(dataset_folder=args.dataset_folder,
                    download_dataset_folder=args.download_folder,
                    test_scenario=args.test_scenario,
                    test_topology=args.test_topology,
                    frequencies=args.frequencies,
                    simulation_time=args.sim_time,
                    time_att_start=args.time_att_start,
                    chosen_detector=args.model,
                    out_path=args.out_folder)
    tester.run()


if __name__ == '__main__':
    args = gather_settings()
    main(args)
