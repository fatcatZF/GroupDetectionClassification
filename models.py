from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from utils import *

_EPS = 1e-10


class MLP(nn.Module):
    """
    Two-Layer fully-connected ELU net with batch norm
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        n_in: #units of input layers
        n_hid: #units of hidden layers
        n_out: #units of output layers
        do_prob: dropout probability
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

    def batch_norm(self, inputs):
        """
        inputs.size(0): batch size
        inputs.size(1): number of channels
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    
    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        #print(type(inputs))
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)



class CausalConv1d(nn.Module):
    """
    causal conv1d layer
    return the sequence with the same length after 1D causal convolution
    Input: [n_batch, in_channels, L]
    Output: [n_batch, out_channels, L]
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation*(kernel_size-1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                            padding=self.padding, dilation=dilation)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        if self.kernel_size==1:
            return x
        return x[:,:,:-self.padding]     


class GatedCausalConv1d(nn.Module):
    """
    Gated Causal Conv1d Layer
    h_(l+1)=tanh(Wg*h_l)*sigmoid(Ws*h_l)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(GatedCausalConv1d, self).__init__()
        self.convg = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation) #gate
        self.convs = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation)
        
    def forward(self, x):
        return torch.sigmoid(self.convg(x))*torch.tanh(self.convs(x))





class GatedResCausalConvBlock(nn.Module):
    """
    Gated Residual Convolutional block
    """     
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(GatedResCausalConvBlock, self).__init__()
        self.conv1 = GatedCausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = GatedCausalConv1d(n_out, n_out, kernel_size, dilation*2)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return x




class GatedResCausalCNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, kernel_size=5, depth=2, 
                 do_prob=0.):
        """
        n_in: number of input channels
        n_hid, n_out: number of output channels
        """
        super(GatedResCausalCNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                dilation=1, return_indices=False, ceil_mode=False)
        res_layers = []#residual convolutional layers
        for i in range(depth):
            in_channels = n_in if i==0 else n_hid
            res_layers += [GatedResCausalConvBlock(in_channels, n_hid, kernel_size, 
                                                   dilation=2**(2*i))]
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid,1, kernel_size=1)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        #inputs shape:[batch_size*num_edges, num_dims, num_timesteps]
        x = self.res_blocks(inputs)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        pred = self.conv_predict(x)
        attention = F.softmax(self.conv_attention(x), dim=-1)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob



class EncoderSym(nn.Module):
    """Symmetric encoder"""
    def __init__(self, n_in, edge_dim, n_hid, n_out, kernel_size=5, depth=1, do_prob=0.):
        super(EncoderSym, self).__init__()
        self.dropout_prob = do_prob
        self.cnn = GatedResCausalCNN(edge_dim, n_hid, n_hid, kernel_size, depth, do_prob=0.)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob=0.)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob=0.)
        self.mlp3 = MLP(n_hid*2, n_hid, n_hid, do_prob=0.)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.edge_extractors_temp = nn.ParameterList([Parameter(torch.rand(n_in)) for i in range(edge_dim)])
        self.edge_extractors = nn.ParameterList([Parameter(torch.rand(n_hid)) for i in range(n_hid)])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [n_batch, n_atoms, n_timesteps*n_in]
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [n_batch*n_edges, n_timesteps, n_in]

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                              inputs.size(2), inputs.size(3))
        edges = [(senders*D*receivers).sum(-1)/senders.size(-1) for D in self.edge_extractors_temp]       
        edges = torch.stack(edges).permute(1,0,2)
        #shape: [n_batch*n_edges, edge_dim, n_timesteps]
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = [(senders*D*receivers).sum(-1)/senders.size(-1) for D in self.edge_extractors]
        edges = torch.stack(edges).permute(1,2,0)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # inputs shape: [n_batch, n_atoms, n_timesteps, n_in]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [n_batch*n_edges, edge_dim, n_timesteps]
        x = self.cnn(edges)
        #shape: [n_batch*n_edges, n_hid]
        x = x.view(inputs.size(0), (inputs.size(1)-1)*(inputs.size(1)), -1)
        x = self.mlp1(x) # shape: [n_batch, n_edges, n_hid]
        x_skip = x
        x = self.edge2node(x, rel_rec, rel_send)
        x = self.mlp2(x)
        #shape: [n_batch, n_nodes, n_hid]
        x = self.node2edge(x, rel_rec, rel_send)
        #shape: [n_batch, n_edges, n_hid]
        x = torch.cat([x, x_skip], dim=2)
        x = self.mlp3(x)

        return self.fc_out(x)

    


    




    
