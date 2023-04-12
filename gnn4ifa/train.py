# from gnn4ifa.data import IfaDataset
#
# dataset = IfaDataset(root='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
#                      download_folder='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data')
# print('dataset.raw_dir: {}'.format(dataset.raw_dir))
# print('dataset.processed_dir: {}'.format(dataset.processed_dir))

import argparse
from memory_profiler import profile
import os, platform, subprocess, socket, psutil, netifaces, cpuinfo
# Import my modules
from gnn4ifa.bin import Trainer


def gather_settings():
    parser = argparse.ArgumentParser(description='Settings for training of GNN.')
    parser.add_argument('--download_folder', default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data',
                        help='folder where raw dataset will be downloaded')
    parser.add_argument('--dataset_folder',
                        default='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
                        help='folder where tg dataset will be stored')
    parser.add_argument('--train_scenario', default='existing',
                        help='simulations train_scenario to consider')
    parser.add_argument('--train_topology', default='small',
                        help='simulations train_topology to be used')
    parser.add_argument('--frequencies', nargs="+", type=int, default=None,  # default=[8,12,32,64]
                        help='Frequencies of attacks in the simulations to be used as training')
    parser.add_argument('--train_sims', nargs="+", type=int, default=[1, 2, 3],
                        help='IDs list of simulations to be used as training')
    parser.add_argument('--val_sims', nargs="+", type=int, default=[4],
                        help='IDs list of simulations to be used as validation')
    parser.add_argument('--test_sims', nargs="+", type=int, default=[5],
                        help='IDs list of simulations to be used as test')
    parser.add_argument('--train_freq', type=float, default=-1,
                        help='IDs list of simulations to be used as training')
    parser.add_argument('--val_freq', type=float, default=-1,
                        help='IDs list of simulations to be used as validation')
    parser.add_argument('--test_freq', type=float, default=-1,
                        help='IDs list of simulations to be used as test')
    parser.add_argument('--split_mode', default='file_ids',
                        help='cuda if GPU is needed, cpu otherwise')
    parser.add_argument('--sim_time', type=int, default=300,
                        help='Time of simulations')
    parser.add_argument('--time_att_start', type=int, default=50,
                        help='Attack start time in simulations')
    parser.add_argument('--differential', default=0, type=int,
                        help='set to 1 if differential between node features is used for dataset, else set to 0')
    parser.add_argument('--model', default='class_gcn_2x100_mean',
                        help='model to be used for training')
    parser.add_argument('--masking', default=1, type=int,
                        help='set to 1 if training using node masking technique, else set to 0')
    parser.add_argument('--percentile', default=0.99, type=float,
                        help='percentile to be used during anomaly detection')
    parser.add_argument('--optimizer', default='sgd',
                        help='optimizer to be used during training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of the optimizer (used only in combination with SGD)')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay parameter to be used in training')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='number of graphs to be used in each batch')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to be used in training')
    parser.add_argument('--lr', default=1e-1, type=float,
                        help='learning rate to be used in training')
    parser.add_argument('--out_folder', default='outputs',
                        help='folder where outputs are stored (best trained models, etc.)')
    parser.add_argument('--device', default='cpu',
                        help='cuda if GPU is needed, cpu otherwise')
    args = parser.parse_args()
    return args


def system():
    kb = float(1024)
    mb = float(kb ** 2)
    gb = float(kb ** 3)
    memTotal = int(psutil.virtual_memory()[0] / gb)
    storageTotal = int(psutil.disk_usage('/')[0] / gb)
    info = cpuinfo.get_cpu_info()['brand_raw']
    core = os.cpu_count()
    host = socket.gethostname()
    print()
    print('---------- System Info ----------')
    print()
    print("Hostname     :", host)
    print("System       :", platform.system(), platform.machine())
    print("Kernel       :", platform.release())
    print('Compiler     :', platform.python_compiler())
    print('CPU          :', info, core, "(Core)")
    print("Memory       :", memTotal, "GiB")
    print("Disk         :", storageTotal, "GiB")


@profile
def main(args):
    print(args)
    trainer = Trainer(dataset_folder=args.dataset_folder,
                      download_dataset_folder=args.download_folder,
                      train_scenario=args.train_scenario,
                      train_topology=args.train_topology,
                      frequencies=args.frequencies,
                      train_sim_ids=args.train_sims,
                      val_sim_ids=args.val_sims,
                      test_sim_ids=args.test_sims,
                      train_freq=args.train_freq,
                      val_freq=args.val_freq,
                      test_freq=args.test_freq,
                      split_mode=args.split_mode,
                      simulation_time=args.sim_time,
                      time_att_start=args.time_att_start,
                      differential=True if args.differential == 1 else False,
                      chosen_model=args.model,
                      masking=True if args.masking == 1 else False,
                      percentile=args.percentile,
                      optimizer=args.optimizer,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      lr=args.lr,
                      out_path=args.out_folder)
    trainer.run()


if __name__ == '__main__':
    system()
    args = gather_settings()
    main(args)
