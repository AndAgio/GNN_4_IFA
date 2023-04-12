import torch
import torch.nn.functional as t_func
import torch_geometric as tg


class Classifier(torch.nn.Module):
    def __init__(self,
                 input_node_dim,
                 conv_type='gcn',
                 hidden_dim=100,
                 n_layers=2,
                 pooling_type='mean',
                 n_classes=2):
        assert conv_type in ['gcn', 'gat', 'cheb', 'gin', 'tag', 'sg']
        assert pooling_type in ['mean', 'sum', 'max', 's2s', 'att']
        self.n_layers = n_layers
        super().__init__()
        # Define layers
        self.convs = torch.nn.ModuleList()
        for index in range(n_layers):
            # Define input and output dimension of the convolutional layer
            in_dim = input_node_dim if index == 0 else hidden_dim
            out_dim = hidden_dim
            if conv_type == 'gcn':
                self.convs.append(tg.nn.GCNConv(in_dim,
                                                out_dim))
            elif conv_type == 'gat':
                self.convs.append(tg.nn.GATConv(-1,
                                                out_dim,
                                                heads=4))
            elif conv_type == 'cheb':
                self.convs.append(tg.nn.ChebConv(in_dim,
                                                 out_dim,
                                                 K=3))
            elif conv_type == 'gin':
                self.convs.append(tg.nn.GINConv(nn=torch.nn.Linear(in_dim,
                                                                   out_dim)))
            elif conv_type == 'tag':
                self.convs.append(tg.nn.TAGConv(in_dim,
                                                out_dim,
                                                K=3))
            elif conv_type == 'sg':
                self.convs.append(tg.nn.SGConv(in_dim,
                                               out_dim))
            else:
                raise ValueError('Something went wrong with convolution type {}!'.format(conv_type))
        # Define layer for classifying the whole graph
        if pooling_type == 'mean':
            self.global_pooling = tg.nn.global_mean_pool
        elif pooling_type == 'sum':
            self.global_pooling = tg.nn.global_add_pool
        elif pooling_type == 'max':
            self.global_pooling = tg.nn.global_max_pool
        elif pooling_type == 's2s':
            self.global_pooling = tg.nn.Set2Set(in_channels=hidden_dim,
                                                processing_steps=1,
                                                num_layers=2)
        elif pooling_type == 'att':
            self.global_pooling = tg.nn.GlobalAttention(gate_nn=torch.nn.Linear(hidden_dim, 1),
                                                        nn=torch.nn.Linear(hidden_dim, 2 * hidden_dim))
        else:
            raise ValueError('Something went wrong with pooling type {}!'.format(pooling_type))
        self.class_layer = torch.nn.Linear(hidden_dim if pooling_type not in ['s2s', 'att'] else 2 * hidden_dim,
                                           n_classes)

    def forward(self, data):
        # Extract x and edges from data
        H_n, edge_index, batch = data.x.float(), data.edge_index, data.batch
        # Obtain node embeddings via graph convolutions
        for index in range(self.n_layers):
            # Apply convolution
            H_n = self.convs[index](H_n, edge_index)
            # Apply activations
            H_n = t_func.relu(H_n)
        # Apply global pooling to construct graph representation
        H_n = self.global_pooling(H_n, batch)
        # Apply a classification layer
        H_n = t_func.dropout(H_n, p=0.5, training=self.training)
        out = self.class_layer(H_n)
        return out
