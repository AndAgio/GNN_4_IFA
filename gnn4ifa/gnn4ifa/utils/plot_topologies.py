import glob
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import matplotlib
import warnings


font = {'size': 15}
matplotlib.rc('font', **font)
warnings.filterwarnings("ignore")


def plot_topologies(topologies, save_singles=True):
    files = glob.glob(os.path.join('/', *os.path.abspath(__file__).split('/')[:-1], 'topology_files', '*.txt'))
    print(os.path.join('/', *os.path.abspath(__file__).split('/')[:-1], 'topology_files'))
    print(f'files: {files}')
    topologies_dict = get_data(files)
    print(f'topologies_dict: {topologies_dict}')
    # Iterate over each topology received in input
    n_topos = len(topologies)
    fig, axs = plt.subplots(1, n_topos, figsize=(15, 5))
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'topologies')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Iterate over each topology received in input
    topo_index = 0
    for topology_name, topology_data in topologies_dict.items():
        graph = get_graph_structure(topology_data, debug=False)
        plot_graph(graph, axs[topo_index], node_size=35 if topology_name.split('.')[0] == 'large' else 150)
        axs[topo_index].axis('off')
        title = '{}'.format(topology_name.split('.')[0].upper() if topology_name.split('.')[0] == 'dfn'
                            else topology_name.split('.')[0].capitalize())
        axs[topo_index].title.set_text(title)
        topo_index += 1
    # Save generated graph image
    image_path = os.path.join(out_path, 'topologies.pdf')
    axs[0].legend(scatterpoints=1, loc='center left', bbox_to_anchor=(-0.5, 0.5), prop={'size': 15})
    plt.savefig(image_path)
    plt.show()
    plt.close()
    # Save single images
    if save_singles:
        for topology_name, topology_data in topologies_dict.items():
            graph = get_graph_structure(topology_data, debug=False)
            ax = plt.subplot(111)
            plot_graph(graph, ax, node_size=35 if topology_name.split('.')[0] == 'large' else 150)
            plt.axis('off')
            image_path = os.path.join(out_path, '{}.pdf'.format(topology_name.split('.')[0]))
            plt.legend(scatterpoints=1)
            plt.savefig(image_path)
            plt.show()
            plt.close()


def get_graph_structure(topology_lines_dataframe, debug=False):
    # Open file containing train_topology structure
    topology_lines_dataframe = topology_lines_dataframe.reset_index()  # make sure indexes pair with number of rows
    links = []
    for index, row in topology_lines_dataframe.iterrows():
        source = row['Source']
        dest = row['Destination']
        links.append([source, dest])
    # print('router_links: {}'.format(router_links))
    list_of_nodes = list(set([elem for link in links for elem in link]))
    # Use nx to obtain the graph corresponding to the graph
    graph = nx.Graph()
    # Build the DODAG graph from nodes and edges lists
    graph.add_nodes_from(list_of_nodes)
    graph.add_edges_from(links)
    # Print and plot for debugging purposes
    if debug:
        print('graph: {}'.format(graph))
        print('graph.nodes: {}'.format(graph.nodes))
        print('graph.edges: {}'.format(graph.edges))
        ax = plt.subplot(111)
        plot_graph(graph, ax)
        plt.axis('off')
        plt.show()
        # nx.draw(graph, with_labels=True, pos=pos, font_weight='bold')
        # plt.show()
    # Return networkx graph
    return graph


def plot_graph(graph, ax, node_size=150):
    pos = nx.spring_layout(graph)
    # pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spectral_layout(graph)
    nodes = graph.nodes()
    # print('nodes: {}'.format(nodes))
    colors = [get_color(node) for node in nodes]
    indices = [i for i, node in enumerate(nodes) if node[:4] == 'Rout']
    nx.draw_networkx_nodes(graph, pos, nodelist=[list(nodes)[ind] for ind in indices],
                           node_color=[colors[ind] for ind in indices],
                           node_size=node_size, ax=ax, label='Routers')
    indices = [i for i, node in enumerate(nodes) if node[:4] == 'Prod']
    nx.draw_networkx_nodes(graph, pos, nodelist=[list(nodes)[ind] for ind in indices],
                           node_color=[colors[ind] for ind in indices],
                           node_size=node_size, ax=ax, label='Producers')
    indices = [i for i, node in enumerate(nodes) if node[:4] == 'Cons']
    nx.draw_networkx_nodes(graph, pos, nodelist=[list(nodes)[ind] for ind in indices],
                           node_color=[colors[ind] for ind in indices],
                           node_size=node_size, ax=ax, label='Consumers')
    nx.draw_networkx_edges(graph, pos, alpha=0.9, ax=ax)


def get_color(node_name):
    node_type = node_name[:4]
    node_colors = {'Rout': (175 / 255, 141 / 255, 195 / 255),
                   'Prod': (44 / 255, 123 / 255, 182 / 255),
                   'Cons': (127 / 255, 191 / 255, 123 / 255), }
    # node_colors = {'Rout': (166 / 255, 206 / 255, 227 / 255),
    #                'Prod': (31 / 255, 120 / 255, 180 / 255),
    #                'Cons': (178 / 255, 223 / 255, 138 / 255), }
    return node_colors[node_type]


def random_sample_producers_from_large_topology(files, n_producers=2):
    large_file = [file for file in files if file.split('/')[-1] == 'format-large.txt'][0]
    file_data = pd.read_csv(large_file, sep='\t', index_col=False)
    nodes = list(set(file_data["Source"].tolist() + file_data["Destination"].tolist()))
    consumers = [node for node in nodes if node[:4] == 'Cons']
    selected_consumers = random.sample(consumers, n_producers)
    producers = ['Prod{}'.format(item[4:]) for item in selected_consumers]
    # Replace selected routers in the format file
    fin = open(large_file, "rt")
    fout = open(f'{large_file}_2', "wt")
    for line in fin:
        for check, rep in zip(selected_consumers, producers):
            line = line.replace(check, rep)
        fout.write(line)
    # close input and output files
    fin.close()
    fout.close()
    os.remove(large_file)
    os.rename(f'{large_file}_2', large_file)


def get_data(files):
    # Covert files to decent format
    for file in files:
        file_type = file.split('/')[-1].split('-')[0]
        if file_type != 'format':
            convert_topology_to_decent_format(file)
    files = glob.glob(os.path.join('/', *os.path.abspath(__file__).split('/')[:-1], 'topology_files', '*.txt'))
    random_sample_producers_from_large_topology(files)
    # Define empty dictionary containing data
    data = {}
    # Iterate over all simulation files and read them
    for file in files:
        # print('file: {}'.format(file))
        # Check file type
        file_type = file.split('/')[-1].split('-')[0]
        if file_type == 'format':
            topo_name = file.split('/')[-1].split('-')[1]
            # Read csv file
            file_data = pd.read_csv(file, sep='\t', index_col=False)
            # Put data in dictionary
            data[topo_name] = file_data
        else:
            continue
    return data


def delete_formatted_files():
    files = glob.glob(os.path.join('/', *os.path.abspath(__file__).split('/')[:-1], 'topology_files', '*.txt'))
    for file in files:
        file_type = file.split('/')[-1].split('-')[0]
        if file_type == 'format':
            os.remove(file)


def get_lines_from_unformatted_topology_file(file):
    reversed_lines = reversed(list(open(file)))
    keep = []
    # Cycle through file from bottom to top
    for line in reversed_lines:
        if line.rstrip() == '':
            continue
        if line.rstrip()[0] == '#':
            break
        else:
            keep.append('\t'.join(line.split()) + '\n')
    keep = reversed(keep)
    return keep


def convert_topology_to_decent_format(file):
    final_lines = get_lines_from_unformatted_topology_file(file)
    # Store file with new name
    new_file = os.path.join('/', os.path.join(*file.split('/')[:-1]),
                            'format-{}'.format(file.split('/')[-1]))
    if os.path.exists(new_file):
        os.remove(new_file)
    with open(new_file, 'w') as f:
        f.write('Source\tDestination\n')
        for item in final_lines:
            f.write(item)
    return new_file


def main():
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn', 'large']
    # Run distribution plotter
    plot_topologies(topologies)
    delete_formatted_files()


if __name__ == '__main__':
    main()
