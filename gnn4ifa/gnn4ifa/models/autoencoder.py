import torch
import torch.nn.functional as t_func
import torch_geometric as tg


class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 input_node_dim,
                 conv_type='gcn',
                 hidden_dim=100,
                 n_encoding_layers=2,
                 n_decoding_layers=2):
        super(AutoEncoder, self).__init__()
        assert conv_type in ['gcn', 'gat', 'cheb', 'gin', 'tag', 'sg']
        self.n_encoding_layers = n_encoding_layers
        self.n_decoding_layers = n_decoding_layers
        print(f'n_encoding_layers: {n_encoding_layers}')
        print(f'n_decoding_layers: {n_decoding_layers}')
        print(f'hidden_dim: {hidden_dim}')
        # Define layers for encoding
        self.encoding_convs = torch.nn.ModuleList()
        for index in range(n_encoding_layers):
            # Define input and output dimension of the convolutional layer
            in_dim = input_node_dim if index == 0 else hidden_dim
            out_dim = hidden_dim
            if conv_type == 'gcn':
                self.encoding_convs.append(tg.nn.GCNConv(in_dim,
                                                         out_dim))
            elif conv_type == 'gat':
                self.encoding_convs.append(tg.nn.GATConv(-1,
                                                         out_dim,
                                                         heads=4))
            elif conv_type == 'cheb':
                self.encoding_convs.append(tg.nn.ChebConv(in_dim,
                                                          out_dim,
                                                          K=3))
            elif conv_type == 'gin':
                self.encoding_convs.append(tg.nn.GINConv(nn=torch.nn.Linear(in_dim,
                                                                            out_dim)))
            elif conv_type == 'tag':
                self.encoding_convs.append(tg.nn.TAGConv(in_dim,
                                                         out_dim,
                                                         K=3))
            elif conv_type == 'sg':
                self.encoding_convs.append(tg.nn.SGConv(in_dim,
                                                        out_dim))
            else:
                raise ValueError('Something went wrong with convolution type {}!'.format(conv_type))
        # Define layers for decoding
        self.decoding_convs = torch.nn.ModuleList()
        for index in range(n_decoding_layers):
            # Define input and output dimension of the convolutional layer
            in_dim = hidden_dim
            out_dim = input_node_dim if index == n_decoding_layers - 1 else hidden_dim
            if conv_type == 'gcn':
                self.decoding_convs.append(tg.nn.GCNConv(in_dim,
                                                         out_dim))
            elif conv_type == 'gat':
                self.decoding_convs.append(tg.nn.GATConv(-1,
                                                         out_dim,
                                                         heads=4))
            elif conv_type == 'cheb':
                self.decoding_convs.append(tg.nn.ChebConv(in_dim,
                                                          out_dim,
                                                          K=3))
            elif conv_type == 'gin':
                self.decoding_convs.append(tg.nn.GINConv(nn=torch.nn.Linear(in_dim,
                                                                            out_dim)))
            elif conv_type == 'tag':
                self.decoding_convs.append(tg.nn.TAGConv(in_dim,
                                                         out_dim,
                                                         K=3))
            elif conv_type == 'sg':
                self.decoding_convs.append(tg.nn.SGConv(in_dim,
                                                        out_dim))
            else:
                raise ValueError('Something went wrong with convolution type {}!'.format(conv_type))

    def encode(self, H_n, edge_index, detach=False):
        # Apply layers of graph convolution to update nodes embedding
        H_n, edge_index = self.pass_through(H_n, edge_index, mode='encoder')
        if detach:
            H_n, edge_index = H_n.detach().cpu(), edge_index.detach().cpu()
        return H_n, edge_index

    def decode(self, H_n, edge_index):
        H_n, edge_index = self.pass_through(H_n, edge_index, mode='decoder')
        return H_n, edge_index

    def pass_through(self, H_n, edge_index, mode='encoder'):
        assert mode in ['encoder', 'decoder']
        convs = self.encoding_convs if mode == 'encoder' else self.decoding_convs
        for index in range(self.n_encoding_layers if mode == 'encoder' else self.n_decoding_layers):
            # Apply convolutions
            H_n = convs[index](H_n, edge_index)
            # Apply activations
            H_n = t_func.relu(H_n)
        return H_n, edge_index

    def forward(self, data):
        H_n, edge_index, batch = data.x.float(), data.edge_index, data.batch
        # Encode data
        H_n, edge_index = self.encode(H_n, edge_index)
        # Decode data
        H_n, edge_index = self.decode(H_n, edge_index)
        # Return obtained embeddings
        return H_n
