import os
import pandas as pd
import glob


def get_lines_from_unformatted_topology_file(file):
    return list(open(file))


def get_attackers_names(file):
    data = file_data = pd.read_csv(file, sep='\t', index_col=False)
    # Get names of transmitter devices
    attackers_names = data['Node'].unique()
    # Consider routers only
    attackers_names = [i for i in attackers_names if 'Atta' in i]
    # print('attackers_names: {}'.format(attackers_names))
    return attackers_names


def get_user_names(lines):
    users_names = []
    for line in lines:
        if 'User-' in line:
            split_line = line.split('\t')
            user_token_positions = [i for i in range(len(split_line)) if 'User-' in split_line[i]]
            for pos in user_token_positions:
                user_name = line.split('\t')[pos]
                print('user_name: {} found in line: {}'.format(user_name, line))
                if user_name not in users_names:
                    users_names.append(user_name)
    # print('users_names: {}'.format(users_names))
    return users_names


def write_new_topo_file(old_topo_file, new_lines):
    # Store file with new name
    new_file = os.path.join('/', os.path.join(*old_topo_file.split('/')[:-1]),
                            'new-{}'.format(old_topo_file.split('/')[-1]))
    if os.path.exists(new_file):
        os.remove(new_file)
    with open(new_file, 'w') as f:
        for item in new_lines:
            if item == 'Source\tDestination\n':
                continue
            f.write(item)
    return new_file


def simulations_path(download_folder, scenario, topology):
    return os.path.join(os.getcwd(), '../../..', download_folder,
                        'IFA_4_{}'.format(scenario) if scenario != 'normal' else scenario,
                        '{}_topology'.format(topology) if topology != 'dfn' else '{}_topology'.format(topology.upper()))


def convert_topology_files(download_folder):
    path = simulations_path(download_folder, 'existing', 'large')
    # Get files list containing files of current scenario
    files_list = glob.glob(os.path.join(path, '*', '*', '*', '*.txt'))

    for file in files_list:
        # Extract data from the considered simulation
        file_type = file.split('/')[-1].split('-')[0]
        if file_type == 'topology':
            # print('Converting topology file to decent format...')
            folder = os.path.join('/', os.path.join(*file.split('/')[:-1]))
            original_topo_file = file
            corresponding_rate_file = 'rate-trace-{}'.format(file.split('/')[-1].split('-')[-1])
            corresponding_rate_file = os.path.join(folder, corresponding_rate_file)
            # print('topology_file: {}'.format(original_topo_file))
            # print('corresponding_rate_file: {}'.format(corresponding_rate_file))
            attackers_names = get_attackers_names(corresponding_rate_file)
            attackers_indices = [int(att_name.split('Atta')[-1]) for att_name in attackers_names]
            # print('attackers_names: {}'.format(attackers_names))
            # print('attackers_indices: {}'.format(attackers_indices))

            topo_file_lines = get_lines_from_unformatted_topology_file(original_topo_file)
            # print('topo_file_lines: {}'.format(topo_file_lines))
            user_names = get_user_names(topo_file_lines)
            name_mapping = {}
            for index, user_name in enumerate(user_names):
                name_mapping[user_name] = 'Atta{}'.format(index + 1) if \
                    index + 1 in attackers_indices else \
                    'Cons{}'.format(index + 1)
            # print('name_mapping: {}'.format(name_mapping))

            new_topo_file_lines = []
            for line in topo_file_lines:
                new_line = line
                if 'User-' in line:
                    split_line = line.split('\t')
                    user_token_positions = [i for i in range(len(split_line)) if 'User-' in split_line[i]]
                    for pos in user_token_positions:
                        user_name = line.split('\t')[pos]
                        new_line = new_line.replace(user_name, name_mapping[user_name])
                new_topo_file_lines.append(new_line)

            new_topo_file = new_file = os.path.join('/', os.path.join(*original_topo_file.split('/')[:-1]),
                                                    'new-{}'.format(original_topo_file.split('/')[-1]))
            write_new_topo_file(original_topo_file, new_topo_file_lines)

            os.rename(original_topo_file, os.path.join('/', os.path.join(*original_topo_file.split('/')[:-1]),
                                                       'original-{}'.format(original_topo_file.split('/')[-1])))
            os.rename(new_topo_file, original_topo_file)
            os.remove(os.path.join('/', os.path.join(*original_topo_file.split('/')[:-1]),
                                   'original-{}'.format(original_topo_file.split('/')[-1])))


def main():
    # Define scenarios for which the distribution plot is required
    download_folder = 'ifa_data'
    # Run distribution plotter
    convert_topology_files(download_folder)


if __name__ == '__main__':
    main()
