import os
import numpy as np
import pandas as pd
import glob
import pickle
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 22}
matplotlib.rc('font', **font)


def plot_conv_comparison_uad(files_list, topologies, convs, percentile):
    out_path = get_out_path()
    for topo in topologies:
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('{}'.format(percentile))
        # for file in files:
        #     print('file.split(\'_\')[-6]: {}'.format(file[0].split('_')[-6]))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        files = [file for file in files_list
                 if topo in file and
                 file.split('_')[-2].split('=')[-1] == percentile and
                 file.split('_')[-6] in convs]
        # print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
        tps = {}
        fps = {}
        for conv in convs:
            file = [file for file in files if conv in file][0]
            print('conv is: {} -> file found is: {}'.format(conv, file.split('/')[-1]))
            with open(file, "rb") as input_file:
                data = pickle.load(input_file)
            tps[conv] = data['tps']['avg']['avg']
            fps[conv] = data['fps']['avg']['avg']
        results = {'tps': tps, 'fps': fps}
        # tps = pd.DataFrame(tps, index=[0])
        # fps = pd.DataFrame(fps, index=[0])
        results = pd.DataFrame(results)
        results.index = results.index.map(str.upper)
        print('results: {}'.format(results))
        fig = plt.figure()  # Create matplotlib figure
        ax1 = fig.add_subplot(111)  # Create matplotlib axes
        ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.
        width = 0.4
        results.tps.plot(kind='bar', color='tomato', ax=ax1, width=width, position=1, rot=45, align='center')
        results.fps.plot(kind='bar', color='royalblue', ax=ax2, width=width, position=0, align='center')
        ax1.set_ylabel('TPR (%)')
        ax1.set_ylim(0, 100)
        ax1.set_xlim(-0.5, 4.5)
        ax2.set_ylabel('FPR (%)')
        ax2.set_ylim(0, 15)
        ax2.set_xlim(-0.5, 4.5)
        plt.tight_layout()
        image_name = 'UAD_convs_{}'.format(topo)
        image_path = os.path.join(out_path, image_name)
        plt.savefig(image_path + '.pdf')
        plt.show()


def plot_percentile_comparison(files_list, topologies, conv, percentiles):
    for topo in topologies:
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('{}'.format(percentiles))
        # for file in files:
        #     print('file.split(\'_\')[-6]: {}'.format(file[0].split('_')[-6]))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        files = [file for file in files_list
                 if topo in file and
                 file.split('_')[-2].split('=')[-1] in percentiles and
                 file.split('_')[-6] == conv]
        print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
        tps = {}
        fps = {}
        for percentile in percentiles:
            file = [file for file in files if file.split('_')[-2].split('=')[-1] == percentile][0]
            print('percentile is: {} -> file found is: {}'.format(percentile, file.split('/')[-1]))
            with open(file, "rb") as input_file:
                data = pickle.load(input_file)
            tps[percentile] = data['tps']['avg']['avg']
            fps[percentile] = data['fps']['avg']['avg']
        fig = plt.figure(figsize=(15, 8))  # Create matplotlib figure
        ax1 = fig.add_subplot(111)  # Create matplotlib axes
        ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.
        print([float(v) for v in list(tps.keys())])
        ax1.plot([float(v) for v in list(tps.keys())], list(tps.values()), color=conv_color(conv), linestyle='-',
                 linewidth=3, label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))
        ax2.plot([float(v) for v in list(fps.keys())], list(fps.values()), color=conv_color(conv), linestyle='-.',
                 linewidth=3, label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))
        ax1.set_ylabel('TPR (%)')
        ax1.set_ylim(0, 110)
        ax2.set_ylabel('FPR (%)')
        ax2.set_ylim(0, 15)
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.tight_layout()
        plt.show()


def plot_micro_percentile_comparison(files_list, topologies, convs, percentiles):
    out_path = get_out_path()
    fig, axs = plt.subplots(1, 2 * len(topologies), figsize=(20, 5))  # Create matplotlib figure
    for topo_index, topo in enumerate(topologies):
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('{}'.format(percentiles))
        # for file in files:
        #     print('file.split(\'_\')[-6]: {}'.format(file[0].split('_')[-6]))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        lines = []
        for conv in convs:
            files = [file for file in files_list
                     if topo in file and
                     file.split('_')[-2].split('=')[-1] in percentiles and
                     file.split('_')[-6] == conv]
            print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
            tps = {}
            fps = {}
            for percentile in percentiles:
                file = [file for file in files if file.split('_')[-2].split('=')[-1] == percentile][0]
                print('percentile is: {} -> file found is: {}'.format(percentile, file.split('/')[-1]))
                with open(file, "rb") as input_file:
                    data = pickle.load(input_file)
                tps[percentile] = data['tps']['avg']['avg']
                fps[percentile] = data['fps']['avg']['avg']
            print([float(v) for v in list(tps.keys())])
            lines.append(axs[topo_index * 2].plot([float(v) for v in list(tps.keys())], list(tps.values()),
                                                  color=conv_color(conv), marker='o',
                                                  markersize=5, label='{}'.format(
                    conv.upper() if conv != 'cheb' else conv.capitalize()))[0])
            axs[topo_index * 2 - 1].plot([float(v) for v in list(fps.keys())], list(fps.values()),
                                         color=conv_color(conv), marker='o',
                                         markersize=5,
                                         label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))
        axs[topo_index * 2].set_ylabel('TPR (%)')
        axs[topo_index * 2].set_xlabel('Percentile')
        axs[topo_index * 2].set_ylim(0, 110)
        axs[topo_index * 2 - 1].set_ylabel('FPR (%)')
        axs[topo_index * 2 - 1].set_xlabel('Percentile')
        axs[topo_index * 2 - 1].set_ylim(0, 25)
    print('lines: {}'.format(lines))
    labels = [l.get_label() for l in lines]
    plt.tight_layout()
    lgd = plt.legend(lines, labels, loc='center left', bbox_to_anchor=(-3.25, -0.4), ncol=len(convs))
    plt.tight_layout()
    image_name = 'UAD_percentile'
    image_path = os.path.join(out_path, image_name)
    plt.savefig(image_path + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_conv_comparison_sad(files_list, topologies, convs, pool, mu):
    out_path = get_out_path()
    for topo in topologies:
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('pool: {}'.format(pool))
        print('mu: {}'.format(mu))
        files = [file for file in files_list
                 if topo in file and
                 file.split('_')[-3].split('x')[0] == str(mu) and
                 file.split('_')[-2] == pool and
                 file.split('_')[-4] in convs]
        print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
        accs = {}
        f1s = {}
        for conv in convs:
            print('Looking for conv {}'.format(conv))
            file = [file for file in files if conv in file][0]
            print('conv is: {} -> file found is: {}'.format(conv, file.split('/')[-1]))
            with open(file, "rb") as input_file:
                data = pickle.load(input_file)
            accs[conv] = data['acc']['avg']['avg']
            f1s[conv] = data['f1']['avg']['avg']
        results = {'acc': accs, 'f1': f1s}
        # tps = pd.DataFrame(tps, index=[0])
        # fps = pd.DataFrame(fps, index=[0])
        results = pd.DataFrame(results)
        results.index = results.index.map(str.upper)
        print('results: {}'.format(results))
        fig = plt.figure()  # Create matplotlib figure
        ax1 = fig.add_subplot(111)  # Create matplotlib axes
        ax2 = ax1.twinx()  # Create another axes that shares the same x-axis as ax.
        width = 0.4
        results.acc.plot(kind='bar', color='tomato', ax=ax1, width=width, position=1, rot=45, align='center')
        results.f1.plot(kind='bar', color='royalblue', ax=ax2, width=width, position=0, align='center')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 110)
        ax1.set_xlim(-0.5, 4.5)
        ax2.set_ylabel('F1-score (%)')
        ax2.set_ylim(0, 110)
        ax2.set_xlim(-0.5, 4.5)
        plt.tight_layout()
        image_name = 'SAD_convs'
        image_path = os.path.join(out_path, image_name)
        plt.savefig(image_path + '.pdf')
        plt.show()


def plot_micro_mu_comparison(files_list, topologies, convs, pool, mus):
    out_path = get_out_path()
    fig, axs = plt.subplots(1, 2*len(topologies), figsize=(20, 5))  # Create matplotlib figure
    for topo_index, topo in enumerate(topologies):
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('mus: {}'.format(mus))
        # for file in files:
        #     print('file.split(\'_\')[-6]: {}'.format(file[0].split('_')[-6]))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        lines = []
        for conv in convs:
            files = [file for file in files_list
                     if topo in file and
                     file.split('_')[-3].split('x')[0] in [str(mu) for mu in mus] and
                     file.split('_')[-2] == pool and
                     file.split('_')[-4] == conv]
            print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
            accs = {}
            f1s = {}
            for mu in mus:
                file = [file for file in files if file.split('_')[-3].split('x')[0] == str(mu)][0]
                print('mu is: {} -> file found is: {}'.format(mu, file.split('/')[-1]))
                with open(file, "rb") as input_file:
                    data = pickle.load(input_file)
                accs[mu] = data['acc']['avg']['avg']
                f1s[mu] = data['f1']['avg']['avg']
            print([int(v) for v in list(accs.keys())])
            lines.append(axs[topo_index * 2].plot([int(v) for v in list(accs.keys())], list(accs.values()),
                                     color=conv_color(conv), marker='o',
                                     markersize=5,
                                     label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))[0])
            axs[topo_index * 2-1].plot([int(v) for v in list(f1s.keys())], list(f1s.values()),
                        color=conv_color(conv), marker='o',
                        markersize=5, label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))
        axs[topo_index * 2].set_ylabel('Accuracy (%)')
        axs[topo_index * 2].set_xlabel(r'$\mu$')
        axs[topo_index * 2].set_ylim(0, 110)
        axs[topo_index * 2].set_xlim(0.5, 6.5)
        axs[topo_index * 2-1].set_ylabel('F1-score (%)')
        axs[topo_index * 2-1].set_xlabel(r'$\mu$')
        axs[topo_index * 2-1].set_ylim(0, 110)
        axs[topo_index * 2-1].set_xlim(0.5, 6.5)
    print('lines: {}'.format(lines))
    labels = [l.get_label() for l in lines]
    plt.tight_layout()
    lgd = plt.legend(lines, labels, loc='center left', bbox_to_anchor=(-3.1, -0.4), ncol=len(convs))
    plt.tight_layout()
    image_name = 'SAD_mus'
    image_path = os.path.join(out_path, image_name)
    plt.savefig(image_path + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_micro_pools_comparison(files_list, topologies, convs, pools, mu):
    out_path = get_out_path()
    fig, axs = plt.subplots(1, 2*len(topologies), figsize=(20, 5))  # Create matplotlib figure
    for topo_index, topo in enumerate(topologies):
        print('files path: {}'.format(os.path.join(*files_list[0].split('/')[:-1])))
        print('files: {}'.format([f.split('/')[-1] for f in files_list]))
        print('topo: {}'.format(topo))
        print('pools: {}'.format(pools))
        # for file in files:
        #     print('file.split(\'_\')[-6]: {}'.format(file[0].split('_')[-6]))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        #     print('topo in file: {}'.format(topo in file))
        lines = []
        for conv in convs:
            files = [file for file in files_list
                     if topo in file and
                     file.split('_')[-2] in pools and
                     file.split('_')[-3].split('x')[0] == str(mu) and
                     file.split('_')[-4] == conv]
            print('filtered files: {}'.format([f.split('/')[-1] for f in files]))
            accs = {}
            f1s = {}
            for pool in pools:
                file = [file for file in files if file.split('_')[-2] == pool][0]
                print('pool is: {} -> file found is: {}'.format(pool, file.split('/')[-1]))
                with open(file, "rb") as input_file:
                    data = pickle.load(input_file)
                accs[pool] = data['acc']['avg']['avg']
                f1s[pool] = data['f1']['avg']['avg']
            print([v for v in list(accs.keys())])
            lines.append(axs[topo_index * 2].plot([v for v in list(accs.keys())], list(accs.values()),
                                     color=conv_color(conv), marker='o', markersize=5,  # linestyle='-', linewidth=3,
                                     label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))[0])
            axs[topo_index * 2-1].plot([v for v in list(f1s.keys())], list(f1s.values()),
                        color=conv_color(conv), marker='o', markersize=5,  # linestyle='-', linewidth=3, linewidth=3,
                        label='{}'.format(conv.upper() if conv != 'cheb' else conv.capitalize()))
        axs[topo_index * 2].set_ylabel('Accuracy (%)')
        axs[topo_index * 2].set_xlabel('Pooling layer')
        axs[topo_index * 2].set_ylim(0, 110)
        axs[topo_index * 2].set_xlim(-0.5, 4.5)
        axs[topo_index * 2].tick_params(labelrotation=45)
        axs[topo_index * 2-1].set_ylabel('F1-score (%)')
        axs[topo_index * 2-1].set_xlabel('Pooling layer')
        axs[topo_index * 2-1].set_ylim(0, 110)
        axs[topo_index * 2-1].set_xlim(-0.5, 4.5)
        axs[topo_index * 2-1].tick_params(labelrotation=45)
    print('lines: {}'.format(lines))
    labels = [l.get_label() for l in lines]
    plt.tight_layout()
    lgd = plt.legend(lines, labels, loc='center left', bbox_to_anchor=(-3.1, -0.6), ncol=len(convs))
    plt.tight_layout()
    image_name = 'SAD_pools'
    image_path = os.path.join(out_path, image_name)
    plt.savefig(image_path + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def conv_color(conv):
    conv_colors = {'gcn': 'green',
                   'cheb': 'royalblue',
                   'gin': 'tomato',
                   'tag': 'orange',
                   'sg': 'violet'}
    return conv_colors[conv]


def get_out_path():
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'ablation')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def get_files_list(model='uad'):
    file_names = glob.glob(os.path.join(os.getcwd(), 'outputs', 'results', model.upper(), 'performance', '*.pkl'))
    return file_names


def main():
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn']
    convs = ["gcn", "cheb", "gin", "tag", "sg"]
    percentiles = ["0.9", "0.905", "0.91", "0.915", "0.92", "0.925", "0.93", "0.935", "0.94", "0.945", "0.95", "0.955",
                   "0.96", "0.965", "0.97", "0.975", "0.98", "0.985", "0.99", "0.995", "1.0"]
    # percentiles = np.linspace(0.90, 1, 21).tolist()
    print('percentiles: {}'.format(percentiles))
    # Run plotters for uad
    files_list = get_files_list(model='uad')
    # plot_conv_comparison_uad(files_list, topologies, convs, percentiles[-3])
    # plot_percentile_comparison(files_list, topologies, convs[0], percentiles)
    plot_micro_percentile_comparison(files_list, topologies, convs, percentiles)
    # Run plotters for sad
    files_list = get_files_list(model='sad')
    pools = ["mean", "sum", "max", "s2s", "att"]
    mus = [1, 2, 3, 4, 5, 6]
    plot_conv_comparison_sad(files_list, topologies, convs, pools[0], mus[2])
    plot_micro_mu_comparison(files_list, topologies, convs, pools[0], mus)
    # plot_micro_pools_comparison(files_list, topologies, convs, pools, mus[2])


if __name__ == '__main__':
    main()
