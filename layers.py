# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
import pgl.math as math1
import math
from utils import generate_segment_id
from paddle.fluid.data_feeder import convert_dtype

class AttentivePooling(nn.Layer):
    def __init__(self, in_dim, dropout):
        super(AttentivePooling, self).__init__()
        self.compute_logits = nn.Sequential(
            nn.Linear(2 * in_dim, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.gru = nn.GRUCell(in_dim, in_dim)
    
    def broadcast_graph_feat(self, graph, feat):
        nids = graph._graph_node_index
        nids_ = paddle.concat([nids[1:], nids[-1:]])
        batch_num_nodes = (nids_-nids)[:-1]
        # print("bbb",batch_num_nodes)
        h_list = []
        for i, k in enumerate(batch_num_nodes):
            h_list += [feat[i].tile([k,1])]
            # print("ccc",(h_list))

        return paddle.concat(h_list)
    
    def forward(self, graph, node_feat, graph_feat):
        graph_feat_broad = self.broadcast_graph_feat(graph, F.relu(graph_feat))
        graph_node_feat = paddle.concat([graph_feat_broad, node_feat], axis=1)
        graph_node_feat = self.compute_logits(graph_node_feat)
        node_a = pgl.math.segment_softmax(graph_node_feat, graph.graph_node_id)
        node_h = self.project_nodes(node_feat)
        context = self.pool(graph, node_h * node_a) ## NOTE
        graph_h, _ = self.gru(context, graph_feat) ## NOTE
        return graph_h 

class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))

class DistRBF(nn.Layer):
    def __init__(self, K, cut_r, requires_grad=False):
        super(DistRBF, self).__init__()
        self.K = K
        self.cut_r = cut_r
        # self.mu = self.create_parameter(paddle.linspace(math.exp(-cut_r), 1., K).unsqueeze(0))
        # self.beta = self.create_parameter(paddle.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2)))
        self.mu = paddle.linspace(math.exp(-cut_r), 1., K).unsqueeze(0)
        self.beta = paddle.full((1, K), math.pow((2 / K) * (1 - math.exp(-cut_r)), -2))
    
    def forward(self, r):
        batch_size = r.size
        K = self.K
        ratio_r = r / self.cut_r
        phi = 1 - 6 * ratio_r.pow(5) + 15 * ratio_r.pow(4) - 10 * ratio_r.pow(3)
        phi = paddle.expand(phi, shape=[batch_size, K])
        local_r = paddle.expand(r, shape=[batch_size, K])
        g = phi * paddle.exp(-self.beta.expand([batch_size, K]) * (paddle.exp(-local_r) - self.mu.expand([batch_size, K]))**2)
        return g
class SpatialInputLayer(nn.Layer):
    """Implementation of Spatial Relation Embedding Module.
    """
    def __init__(self, hidden_dim, cut_dist, activation=F.relu):
        super(SpatialInputLayer, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1, hidden_dim, sparse=True)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)
        self.dist_rbf = DistRBF(128, 5)

    def forward(self, dist_feat):
        dist = self.dist_rbf (dist_feat)

        # dist = paddle.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).astype('int64') - 1
        # eh_emb = self.dist_embedding_layer(dist)
        eh_emb = self.dist_input_layer(dist)
        # eh_emb = paddle.cast(eh_emb, 'float64')
        return eh_emb


class Atom2BondLayer(nn.Layer):
    """Implementation of Node->Edge Aggregation Layer.
    """
    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super(Atom2BondLayer, self).__init__()
        in_dim = atom_dim * 2 + bond_dim
        self.fc_agg = DenseLayer(in_dim, bond_dim, activation=activation, bias=True)

    def agg_func(self, src_feat, dst_feat, edge_feat):
        h_src = src_feat['h']
        h_dst = dst_feat['h']
        h_agg = paddle.concat([h_src, h_dst, edge_feat['h']], axis=-1)
        return {'h': h_agg}

    def forward(self, g, atom_feat, edge_feat):
        msg = g.send(self.agg_func, src_feat={'h': atom_feat}, dst_feat={'h': atom_feat}, edge_feat={'h': edge_feat})
        bond_feat = msg['h']
        # print ('before fc_layers bond_feat.shape: ', bond_feat.shape)
        # print(g.node_feat["coord"])
        bond_feat = self.fc_agg(bond_feat)
        # print ('after fc_layers bond_feat.shape: ', bond_feat.shape)

        return bond_feat


class Bond2AtomLayer(nn.Layer):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, bond_dim, atom_dim, hidden_dim, num_heads, dropout, merge='mean', activation=F.relu):
        super(Bond2AtomLayer, self).__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = nn.Linear(bond_dim, num_heads * hidden_dim)
        # self.lin = nn.Linear(384,  hidden_dim)
        self.dst_fc = nn.Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_edg = self.create_parameter(shape=[num_heads, hidden_dim])
        
        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["attn"] + dst_feat["attn"] + edge_feat['attn']
        # print(src_feat["attn"].shape)
        # print(dst_feat["attn"].shape)
        # print(dst_feat["attn"].shape)




        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)
        # print(msg['h'])
        feature = msg["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_dim])
        feature = feature * alpha
        if self.merge == 'cat':
            feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        if self.merge == 'mean':
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")

        return feature

    def forward(self, g, atom_feat, bond_feat, edge_feat):
        bond_feat = self.feat_drop(bond_feat)
        atom_feat = self.feat_drop(atom_feat)
        edge_feat = self.feat_drop(edge_feat)

        bond_feat = self.src_fc(bond_feat)
        atom_feat = self.dst_fc(atom_feat)
        edge_feat = self.edg_fc(edge_feat)
        bond_feat = paddle.reshape(bond_feat, [-1, self.num_heads, self.hidden_dim])
        atom_feat = paddle.reshape(atom_feat, [-1, self.num_heads, self.hidden_dim])
        edge_feat = paddle.reshape(edge_feat, [-1, self.num_heads, self.hidden_dim])

        attn_src = paddle.sum(bond_feat * self.weight_src, axis=-1)
        attn_dst = paddle.sum(atom_feat * self.weight_dst, axis=-1)
        attn_edg = paddle.sum(edge_feat * self.weight_edg, axis=-1)

        msg = g.send(self.attn_send_func,
                     src_feat={"attn": attn_src, "h": bond_feat},
                     dst_feat={"attn": attn_dst},
                     edge_feat={'attn': attn_edg})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)
        # rst = self.lin(rst)
        if self.activation:
            rst = self.activation(rst)
            # print('b2a')
        return rst

class Atom2AtomLayer(nn.Layer):
    
    def __init__(self, hidden_dim, dropout,  activation=F.relu):
        super(Atom2AtomLayer, self).__init__()
        
        self.Wq = nn.Linear(36, hidden_dim)
        self.Wk = nn.Linear(36, hidden_dim)
        self.Wv = nn.Linear(36, hidden_dim)
        self.activation = activation
        # self.Wq = nn.Linear(hidden_dim, hidden_dim)
        # self.Wk = nn.Linear(hidden_dim, hidden_dim)
        # self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)


        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)       
        
        self.feat_drop = nn.Dropout(p=dropout)
        self.sqrt_dk = paddle.sqrt(paddle.to_tensor([hidden_dim]).astype('float32'))
        self.softmax = nn.Softmax()
        
    def send_func(self,src_feat, dst_feat, edge_feat):
    # # print('ss',src_feat["h"])
    #     Q = self.Wq(src_feat["h"])
    #     K = self.Wk(src_feat["h"])
    #     V = self.Wv(src_feat["h"])
    #     # print(g)
    #     alpha = paddle.matmul(Q, K.T) / self.sqrt_dk
        # alpha = self.softmax(alpha)
        V = src_feat["h"]
        # a= paddle.to_tensor(neibor,dtype=('float32'))
        # alpha = alpha * neibor
        # alpha = self.softmax(alpha)
        # alpha =  self.feat_drop(alpha)
        return {"h":V}
    
    def recv_func(self,msg):
        # alpha = msg.reduce_softmax(msg["alpha"])
        # alpha = self.feat_drop(alpha)
        feature = msg["h"]
        # feature =paddle.matmul(alpha, feature)
        feature = msg.reduce(feature, pool_type="sum")
        feature = feature * alpha
        
        
        # print(feature)
        
        
        
        return feature
        
        
    def forward(self, atom_h):
        # atom_h =  self.feat_drop(atom_h)
        Q = self.Wq(atom_h)
        K = self.Wk(atom_h)
        V = self.Wv(atom_h)
        # atom_h =  self.feat_drop(atom_h)
        alpha = paddle.matmul(Q, K.T) / self.sqrt_dk
        alpha = self.softmax(alpha)
        # msg = g.send(self.send_func,
        #             src_feat={"h": atom_h},dst_feat={"h": atom_h})
        alpha =  self.feat_drop(alpha)
        # a= paddle.to_tensor(neibor,dtype=('float32'))
        # alpha = alpha * neibor
        # alpha = self.softmax(alpha)
        # alpha =  self.feat_drop(alpha)
        rst = paddle.matmul(alpha, V)
        # rst = g.recv(reduce_func=self.recv_func, msg=msg,alpha)
        d = self.Wq(atom_h)
        h = d+ rst
        h = self.layernorm(h)
        hin2 = h
        h = self.W1(h)
        h = F.relu(h)
        h = self.feat_drop(h)
        h = self.W2(h)
        h = h +hin2
        h = self.layernorm(h)
        return h
    
    
    
    
    
class DomainAttentionLayer(nn.Layer):
    """Implementation of Angle Domain-speicific Attention Layer.
    """
    def __init__(self, bond_dim, hidden_dim, dropout, activation=F.relu):
        super(DomainAttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(2 * bond_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1, bias_attr=False)

        self.a = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(axis=0)

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.activation = activation
    
    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        h_c = paddle.concat([src_feat['h'], src_feat['h']], axis=-1)
        h_c = self.attn_fc(h_c)
        h_c = self.tanh(h_c)
        h_s = self.attn_out(h_c)
        return {"alpha": h_c, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = self.attn_drop(alpha) # [-1, 1]
        feature = msg["h"] # [-1, hidden_dim]
        feature = feature * alpha
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, g, bond_feat):
        bond_feat = self.feat_drop(bond_feat)
        msg = g.send(self.attn_send_func,
                    src_feat={"h": bond_feat},
                    dst_feat={"h": bond_feat})
        rst = g.recv(reduce_func=self.attn_recv_func, msg=msg)

        # print(rst)
        b = paddle.ones((128,1))
        
        alpha = paddle.matmul(bond_feat,b)
        alpha = self.softmax(alpha)
        alpha = self.attn_drop(alpha) 
        sfeat = alpha * bond_feat
        rst = sfeat + rst



        if self.activation:
            rst = self.activation(rst)
            print('fff')
        return rst

class Bond2BondLayer(nn.Layer):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=None):
        super(Bond2BondLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.LayerList()
#         print ('bond_dim: ', bond_dim)
        for _ in range(num_angle):
            conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation
    
    def forward(self, g_list, bond_feat):
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = paddle.concat(h_list, axis=-1)
        if self.merge == 'mean':
            feat_h = paddle.mean(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'sum':
            feat_h = paddle.sum(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'max':
            feat_h = paddle.max(paddle.stack(h_list, axis=-1), axis=1)
        if self.merge == 'cat_max':
            feat_h = paddle.stack(h_list, axis=-1)
            feat_max = paddle.max(feat_h, dim=1)[0]
            feat_max = paddle.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = paddle.reshape(feat_h * feat_max, [-1, self.num_angle * self.hidden_dim])
        # print ('feat_h.shape', feat_h.shape)
        if self.activation:
            feat_h = self.activation(feat_h)
        return feat_h
    

class PiPoolLayer(nn.Layer):
    """Implementation of Pairwise Interactive Pooling Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle):
        super(PiPoolLayer, self).__init__()
        self.bond_dim = bond_dim
        self.num_angle = num_angle
        self.num_type = 4 * 9
        # fc_in_dim = num_angle * bond_dim
        fc_in_dim =  bond_dim
        self.fc_1 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu, bias=True)
        self.fc_2 = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
    
    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        Input example:
            bond_types_batch: [0,0,2,0,1,2] + [0,0,2,0,1,2] + [2]
            type_count_batch: [[3, 3, 0], [1, 1, 0], [2, 2, 1]] # [num_type, batch_size]
        """
        # bond_feat = self.fc_1(paddle.reshape(bond_feat, [-1, self.num_angle*self.bond_dim]))
        bond_feat = self.fc_1(bond_feat)
        # print(bond_feat.shape)
        inter_mat_list =[]
        for type_i in range(self.num_type):
            # print ('len', len(bond_feat))
            type_i_index = paddle.masked_select(paddle.arange(len(bond_feat)), bond_types_batch==type_i)
            if paddle.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(paddle.to_tensor(np.array([0.]*len(type_count_batch[type_i])), dtype='float32'))
                continue
            bond_feat_type_i = paddle.gather(bond_feat, type_i_index)
            graph_bond_index = op.get_index_from_counts(type_count_batch[type_i])
            # graph_bond_id = generate_segment_id_from_index(graph_bond_index)
            graph_bond_id = generate_segment_id(graph_bond_index)
            graph_feat_type_i = math1.segment_pool(bond_feat_type_i, graph_bond_id, pool_type='sum')
            mat_flat_type_i = self.fc_2(graph_feat_type_i).squeeze(1)

            # print(graph_bond_id)
            # print(graph_bond_id.shape, graph_feat_type_i.shape, mat_flat_type_i.shape)
            my_pad = nn.Pad1D(padding=[0, len(type_count_batch[type_i])-len(mat_flat_type_i)], value=-1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = paddle.stack(inter_mat_list, axis=1) # [batch_size, num_type]
        inter_mat_mask = paddle.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = paddle.where(type_count_batch.transpose([1, 0])>0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch

# class OutputLayer(nn.Layer):
#     """Implementation of Prediction Layer.
#     """
#     def __init__(self, atom_dim, hidden_dim_list):
#         super(OutputLayer, self).__init__()
#         self.pool = pgl.nn.GraphPool(pool_type='sum')
#         self.mlp = nn.LayerList()
#         for hidden_dim in hidden_dim_list:
#             self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
#             atom_dim = hidden_dim
#         self.output_layer = nn.Linear(atom_dim, 1)
    
#     def forward(self, g, atom_feat):
#         # print('atom_feat',atom_feat)
#         graph_feat = self.pool(g, atom_feat)
#         # print('graph_feat',graph_feat)
#         for layer in self.mlp:
#             graph_feat = layer(graph_feat)
#         output = self.output_layer(graph_feat)
#         # print('output',output)
#         return output

class OutputLayer(nn.Layer):
    """Implementation of Prediction Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list):
        super(OutputLayer, self).__init__()
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
    
    def forward(self, g, atom_feat):
        # print('atom_feat',atom_feat)
        graph_feat = self.pool(g, atom_feat)
        # graph_feat =  atom_feat
        # print('graph_feat',graph_feat)
        # print("ff",atom_feat.shape)

        for layer in self.mlp:
            graph_feat = layer(graph_feat)
            # print("ffddd",graph_feat.shape)
        output = self.output_layer(graph_feat)
        # print('output',output)
        return output


class TransformerConv(nn.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation='relu'):
        super(TransformerConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.e = nn.Linear(hidden_size, num_heads * hidden_size,bias_attr = False)
        self.w1 = nn.Linear(num_heads * hidden_size,num_heads * hidden_size*2)
        self.w2 = nn.Linear(num_heads * hidden_size*2,num_heads * hidden_size)



        self.q = nn.Linear(input_size, num_heads * hidden_size,bias_attr = False)
        self.k = nn.Linear(input_size, num_heads * hidden_size,bias_attr = False)
        self.v = nn.Linear(input_size, num_heads * hidden_size,bias_attr = False)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

        if skip_feat:
            if concat:
                self.skip_feat = nn.Linear(input_size, num_heads * hidden_size)
            else:
                self.skip_feat = nn.Linear(input_size, hidden_size)
        else:
            self.skip_feat = None

        if gate:
            if concat:
                self.gate = nn.Linear(3 * num_heads * hidden_size, 1)
            else:
                self.gate = nn.Linear(3 * hidden_size, 1)
        else:
            self.gate = None

        if layer_norm:
            if self.concat:
                self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
            else:
                self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def send_attention(self, src_feat, dst_feat, edge_feat):
        if "edge_feat" in edge_feat:
            alpha = dst_feat["q"] * (src_feat["k"] + edge_feat['edge_feat'])
            # alpha = dst_feat["q"] * (src_feat["k"] * edge_feat['edge_feat'])
            e = alpha
            # src_feat["v"] = src_feat["v"] + edge_feat["edge_feat"]
            src_feat["v"] = src_feat["v"] 
            
            
        else:
            alpha = dst_feat["q"] * src_feat["k"]
        alpha = paddle.sum(alpha, axis=-1)
        # print(alpha)
        # print(edge_feat['edge_feat'])
        return {"alpha": alpha, "v": src_feat["v"]+edge_feat['edge_feat'], "e":e}


    def reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["v"]
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
            # print(feature)
            # e_f = paddle.reshape(msg["e"],
            #                          [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)
        feature = msg.reduce(feature, pool_type="sum")
        return feature


    
    
    # def redu_fff(self, msg):
    #     e_f = paddle.reshape(msg["e"],
    #                                  [-1, self.num_heads * self.hidden_size])
    #     return e_f
        
        
    def send_recv(self, graph, q, k, v, edge_feat):
        q = q / (self.hidden_size**0.5)
        if edge_feat is not None:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q},
                edge_feat={'edge_feat': edge_feat})
        else:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q})

        output = graph.recv(reduce_func=self.reduce_attention, msg=msg)
        e_f = paddle.reshape(msg["e"],
                                     [-1, self.num_heads * self.hidden_size])
        # print(output)
        return output,e_f


    def forward(self, graph, feature, edge_feat):
        if self.feat_drop > 1e-5:
            feature = self.feat_dropout(feature)
        q = self.q(feature)
        k = self.k(feature)
        v = self.v(feature)
        edge_feat1 = self.e(edge_feat)
        
        q = paddle.reshape(q, [-1, self.num_heads, self.hidden_size])
        k = paddle.reshape(k, [-1, self.num_heads, self.hidden_size])
        v = paddle.reshape(v, [-1, self.num_heads, self.hidden_size])
        if edge_feat is not None:
            if self.feat_drop > 1e-5:
                edge_feat1 = self.feat_dropout(edge_feat1)
            edge_feat1 = paddle.reshape(edge_feat1,
                                       [-1, self.num_heads, self.hidden_size])

        output,e_f = self.send_recv(graph, q, k, v, edge_feat=edge_feat1)

        if self.skip_feat is not None:
                        
            skip_feat = self.skip_feat(feature)
            # edge_feat = paddle.reshape(edge_feat,[edge_feat.shape[0],512])
            edge_feat = self.e(edge_feat)
            # print('e',e_f)
            # print(edge_feat)
            e= e_f + edge_feat
            h = skip_feat + output
            h = self.layer_norm(h)
            e = self.layer_norm(e)
            h_in2 = h 
            e_in2 = e 

            e = self.w1(e)
            e = F.relu(e)
            e = self.feat_dropout(e)
            e = self.w2(e)
            h = self.w1(h)
            h = F.relu(h)
            h = self.feat_dropout(h)
            h = self.w2(h)

            e = e + e_in2
            h = h + h_in2
            h = self.layer_norm(h)
            e = self.layer_norm(e)

            # if self.gate is not None:
            #     gate = F.relu(
            #         self.gate(
            #             paddle.concat(
            #                 [skip_feat, output, skip_feat - output], axis=-1)))
            #     output = gate * skip_feat + (1 - gate) * output
            # else:
            #     output = skip_feat + output

        # if self.layer_norm is not None:
        #     output = self.layer_norm(output)

        # if self.activation is not None:
        #     output = self.activation(output)
        return h, e
    
class E_GCL(nn.Layer):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=128, act_fn=nn.ReLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf , hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # layer = nn.Linear(hidden_nf, 1, bias_attr=False)
       
        initializer=paddle.nn.initializer.XavierUniform(fan_in=0.001)
        layer = nn.Linear(hidden_nf, 1, weight_attr = initializer,bias_attr=False)
        # torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())     

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = paddle.concat([source, target, radial], axis=1)
        else:
            out = paddle.concat([source, target, radial, edge_attr], axis=1)
        # print(out.shape)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out   

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        # print('edge_attr',edge_attr)
        # print('row',col)
        # print('x',x)

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        # print('agg',agg)

        if node_attr is not None:
            agg = paddle.concat([x, agg, node_attr], axis=1)
        else:
            agg = paddle.concat([x, agg], axis=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        # print(trans)
        # print(coord.shape[0])
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0])
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        # print('coorddiff',coord_diff)
        radial = paddle.sum(coord_diff**2, 1).unsqueeze(1)
        # print('radial',radial)

        if self.normalize:
            norm = paddle.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        # print('kk',coord_diff)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        # print('e',edge_index)
        # print('r',row)
        # print('c',col)
        # print('jjjj',coord)
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # print('r',radial)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        # print('eeee',edge_feat)
        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        # print('coord',coord)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # h, agg = self.node_model(h, edge_index, b, node_attr)

        return h, coord, edge_attr

class EGNN(nn.Layer):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='gpu:0', act_fn=nn.Silu(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        ''',,'''
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        self.conv_layer = nn.LayerList()
        for i in range(0, n_layers):
        # for _ in range(num_angle):
            conv =E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh)
            self.conv_layer.append(conv)


            # self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
            #                                     act_fn=act_fn, residual=residual, attention=attention,
            #                                     normalize=normalize, tanh=tanh))
        # self.to(self.device)
   
    def forward(self, h, x, edge):
        h = self.embedding_in(h)
        # edge = (g.edges).T
        # for i in range(20):
        # print('edge',g.edges)
        for i in range(0, self.n_layers):
            h, x, _ = self.conv_layer[i](h, edge, x)
        h = self.embedding_out(h)
        # print('xx',x)
        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.shape[1])
    result = paddle.zeros(result_shape)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand([-1, data.shape[1]])
    result =  sc(result, segment_ids, data)
    return result

def sc(x, seg, data):
    x = x
    updates = data
    index = seg
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(updates, index=updates_index)
    res = paddle.scatter_nd_add(x, index, updates)
    return res

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.shape[1])
    # print('data',data)
    segment_ids = segment_ids.unsqueeze(-1).expand([-1, data.shape[1]])
    # print('id',segment_ids)

    result =paddle.zeros(result_shape) 
    # print('res',result) # Init empty result tensor.
    count = paddle.zeros(result_shape)
    result = sc(result, segment_ids, data)
    # print('res',result)
    count = sc(count, segment_ids, paddle.ones_like(data))
    # print('ccc',count)
    # print( 'clip',count.clip(min=1))
    # print('result / count.clip(min=1)',result / count.clip(min=1))
    return result / count.clip(min=1)


class CosineCutoff(nn.Layer):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                paddle.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (paddle.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = paddle.to_tensor(cutoffs * (distances < self.cutoff_upper),dtype = "float32")
            return cutoffs
class ExpNormalSmearing(nn.Layer):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        # if trainable:
        #     self.register_parameter("means", nn.Parameter(means))
        #     self.register_parameter("betas", nn.Parameter(betas))
        # else:
        #     self.register_buffer("means", means)
        #     self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = paddle.exp(
            paddle.to_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = paddle.linspace(start_value, 1, self.num_rbf)
        betas = paddle.to_tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        means, betas = self._initial_params()
        # print(betas.shape)
        # means = 
        dist = dist.unsqueeze(-1)
        # print(((paddle.exp(self.alpha * (-dist + self.cutoff_lower)) - means) ** 2).shape)

        return self.cutoff_fn(dist) * paddle.exp(
            -betas.squeeze(1)
            * (paddle.exp(self.alpha * (-dist + self.cutoff_lower)) - means) ** 2
        )

class Distance(nn.Layer):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, edge_index,pos):


        # make sure we didn't miss any neighbors due to max_num_neighbors
        assert not (
            paddle.unique(edge_index[0], return_counts=True)[1] > self.max_num_neighbors
        ).any(), (
            "The neighbor search missed some atoms due to max_num_neighbors being too low. "
            "Please increase this parameter to include the maximum number of atoms within the cutoff."
        )

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        # mask: Optional[torch.Tensor] = None
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = paddle.zeros([edge_vec.shape[0]])
            edge_weight[mask] = paddle.norm(edge_vec[mask], axis=-1)
        else:
            edge_weight = paddle.norm(edge_vec, axis=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        if self.loop and mask is not None:
            # keep self loops even though they might be below the lower cutoff
            lower_mask = lower_mask | ~mask
        # edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return  edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None



class EquivariantMultiHeadAttention(nn.Layer):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__()
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )
        # print(self.propagate)
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = nn.Silu()
        self.aaa = nn.Silu()
        self.bbb = nn.Silu()
        self.attn_activation = nn.Silu()
        self.cutoff = CosineCutoff(0.0, 5.0)
        weight_attr =paddle.nn.initializer.XavierUniform()
        bias_attr = nn.initializer.Constant()
        weight_attr1 =paddle.nn.initializer.XavierUniform()
        bias_attr1 = nn.initializer.Constant()
        weight_attr2 =paddle.nn.initializer.XavierUniform()
        bias_attr2 = nn.initializer.Constant()
        weight_attr3 =paddle.nn.initializer.XavierUniform()
        bias_attr3 = nn.initializer.Constant()
        weight_attr4 =paddle.nn.initializer.XavierUniform()
        bias_attr4 = nn.initializer.Constant()
        weight_attr5 =paddle.nn.initializer.XavierUniform()
        bias_attr5 = nn.initializer.Constant()                        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels,weight_attr=weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels,weight_attr=weight_attr1, bias_attr=bias_attr1)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3,weight_attr=weight_attr2, bias_attr=bias_attr2)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3,weight_attr=weight_attr4, bias_attr=bias_attr4)
        self.lin = nn.Linear(2*self.head_dim, self.head_dim)
        self.lin1 = nn.Linear(self.head_dim, self.head_dim)
        self.vvv = nn.Linear(hidden_channels, 1)
        self.hhh = nn.Linear(128, 128)
        self.ggg = nn.Linear(128, 128)
        self.qq = nn.Linear(128, 128)
        self.kk = nn.Linear(128, 128)
        self.dd = nn.Linear(128, 128)
        self.ff = nn.Linear(128, 128)
        self.rr = nn.Linear(1, 1)
        self.ee = nn.Linear(128, 128)
        self.mmm = nn.Linear(3, 128)
        self.nnn = nn.Linear(3, 128)
        self.jj = nn.Linear(1, 1)
        self.tt = nn.Linear(128, 128)
        self.sss = nn.Softmax()
        self.lin2 = nn.Linear(self.head_dim, 3*self.head_dim)
        self.lin3 = nn.Linear(2*self.head_dim, self.head_dim)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias_attr=False)
        self.drop = nn.Dropout(0.3)
        self.drop1 = nn.Dropout(0.3)
        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels,weight_attr=weight_attr3, bias_attr=bias_attr3)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3,weight_attr=weight_attr5, bias_attr=bias_attr5)
        # self.reset_parameters()
        self.edge_mlp = nn.Sequential(
            nn.Linear(257 , hidden_channels),
            nn.Silu(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Silu())
        initializer=paddle.nn.initializer.XavierUniform(fan_in=0.001)
        layer = nn.Linear(hidden_channels, 1, weight_attr = initializer,bias_attr=False)       

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_channels, hidden_channels))
        coord_mlp.append(nn.Silu())
        coord_mlp.append(layer)

        coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels, hidden_channels),
            nn.Silu(),
            nn.Linear(hidden_channels, hidden_channels))        
    # def reset_parameters(self):
    #     self.layernorm.reset_parameters()
    #     nn.init.xavier_uniform_(self.q_proj.weight)
    #     self.q_proj.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.k_proj.weight)
    #     self.k_proj.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.v_proj.weight)
    #     self.v_proj._(0)
    #     nn.init.xavier_uniform_(self.o_proj.weight)
    #     self.o_proj.bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.vec_proj.weight)
    #     if self.dk_proj:
    #         nn.init.xavier_uniform_(self.dk_proj.weight)
    #         self.dk_proj.bias.data.fill_(0)
    #     if self.dv_proj:
    #         nn.init.xavier_uniform_(self.dv_proj.weight)
    #         self.dv_proj.bias.data.fill_(0)
    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = paddle.concat([source, target, radial], axis=1)
        else:
            out = paddle.concat([source, target, radial, edge_attr], axis=1)
        # print(out.shape)
        out = self.edge_mlp(out)
        return out   
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        # print('coorddiff',coord_diff)
        radial = paddle.sum(coord_diff**2, 1).unsqueeze(1)
        # print('radial',radial)

        # if self.normalize:
        #     norm = paddle.sqrt(radial).detach() + self.epsilon
        #     coord_diff = coord_diff / norm
        # print('kk',coord_diff)
        return radial ,coord_diff
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        # print('edge_attr',edge_attr)
        # print('row',col)
        # print('x',x)

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        # print('agg',agg)

        if node_attr is not None:
            agg = paddle.concat([x, agg, node_attr], axis=1)
        else:
            agg = paddle.concat([x, agg], axis=1)
        out = self.node_mlp(agg)
        out = x + out
        return out
    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        # print(trans)
        # print(coord.shape[0])
        # if self.coords_agg == 'sum':
        #     agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0])
        # elif self.coords_agg == 'mean':
        agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])

        coord = coord + agg
        return coord
    
    def normalize_vector(self,v, dim, eps=1e-6):
        return v / (paddle.linalg.norm(v, p=2, axis=dim, keepdim=True) + eps)
    def project_v2v(self,v, e, dim):
        """
        Description:
            Project vector `v` onto vector `e`.
        Args:
            v:  (N, L, 3).
            e:  (N, L, 3).
        """
        return (e * v).sum(axis=dim, keepdim=True) * e
    def construct_3d_basis(self,center, p1, p2):
        """
        Args:
            center: (N, L, 3), usually the position of C_alpha.
            p1:     (N, L, 3), usually the position of C.
            p2:     (N, L, 3), usually the position of N.
        Returns
            A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
            The matrix is composed of 3 column vectors: [e1, e2, e3].
        """
        v1 = p1 - center    # (N, L, 3)
        e1 = self.normalize_vector(v1, dim=-1)

        v2 = p2 - center    # (N, L, 3)
        u2 = v2 - self.project_v2v(v2, e1, dim=-1)
        e2 = self.normalize_vector(u2, dim=-1)

        e3 = paddle.cross(e1, e2, axis=-1)    # (N, L, 3)

        mat = paddle.concat([
            e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
        ], axis=-1)  # (N, L, 3, 3_index)
        return mat



    def forward(self, x, vec, edge_index, c_ij, f_ij, d_ij,coord,ha,alledge):
        # print("x",x.shape)
        # print("vec",vec.shape)
        # print("r_ij",r_ij.shape)
        # R = self.construct_3d_basis(xxx[0], xxx[1], xxx[2])  # (N, L, 3, 3)
        # qq = paddle.matmul(R, xxx.T)
        # # print(qq)
        # hh0 = self.mmm(qq.T)
        # print(hh0)
        h = self.hhh(ha)
        row, col = edge_index
        edge_attr = None
        node_attr = None
        radial,coord_diff = self.coord2radial(edge_index,coord)
        M_ij = self.edge_model(h[row], h[col], radial,edge_attr)
        h = self.node_model(h, edge_index, M_ij,node_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, M_ij)
        h = self.ggg(h)
        # r_ij = self.vvv(M_ij)
        # r_ij = r_ij.squeeze(1)
        # print("dd",M_ij.shape)
        # print("gg",row)
        r, c = alledge
        bij = self.qq(x[r])*self.kk(x[c])
        cij = self.jj(paddle.norm((coord[r]-coord[c]),2))
        aij = self.sss(bij+cij)
        U = unsorted_segment_sum(aij*self.tt(h[r]), r, num_segments=x.shape[0])

        # print("M_ij",U.shape)
        # print("radial",radial.shape)
        # print("edge_index",edge_index)
        x = self.layernorm(x)
        q = self.q_proj(M_ij).reshape([-1, self.num_heads, self.head_dim])
        k = self.k_proj(M_ij).reshape([-1, self.num_heads, self.head_dim])
        v = self.v_proj(M_ij).reshape([-1, self.num_heads, self.head_dim * 3])
        # print(self.vec_proj(vec))
        vvv = self.vec_proj(vec)
        vec1, vec2, vec3 = paddle.split(vvv, num_or_sections=3, axis=2)
        vec = vec.reshape([-1, 3, self.num_heads, self.head_dim])
        vec_dot = (vec1 * vec2).sum(axis=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim])
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape([-1, self.num_heads, self.head_dim * 3])
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        # # v_j = v[col]
        # out = paddle.concat([q_i, k_j], axis=2)
        # out = self.lin(out)
        # v = self.lin2(out)
        # print("x1",v.shape)
        # print("xqqq",q.shape)

        # v = self.lin2(out)
        # out1 = paddle.concat([q_i, v_j], axis=1)
        x_, vec_ = self.message1(
            edge_index,
            q,
            k,
            v,
            vec,
            dk,
            dv,
            d_ij)

        # print("vec1",vec)
        x_ = x_.reshape([-1, self.hidden_channels])
        vec_ = vec_.reshape([-1, 3, self.hidden_channels])
        # print("dd:",x_.shape)
        # print("fff:",vec_.shape)
        x = unsorted_segment_sum(x_, row, num_segments=x.shape[0])
        v1,v2,v3 = paddle.split(vec_,3,axis = 1)
        # print(v1.shape)
        v1 = unsorted_segment_sum(v1.squeeze(1), row, num_segments=vec.shape[0])
        v2 = unsorted_segment_sum(v2.squeeze(1), row, num_segments=vec.shape[0])
        v3 = unsorted_segment_sum(v3.squeeze(1), row, num_segments=vec.shape[0])
        vec = paddle.stack([v1,v2,v3],axis = 1)


        o1, o2, o3 = paddle.split(self.o_proj(x), 3, axis=1)
        dx = vec_dot * o2 + o3*U
        dvec = vec3 * o1.unsqueeze(1) + vec
        # dx = self.aaa(dx)
        dh = vec_dot*o2+h



        # dvec = self.bbb(dvec)
        return dx, dvec,dh,coord

    def message1(self, edge_index,q,k,v_j, vec_j, dk, dv, d_ij):
        # attention mechanism
        # out1 = self.lin1(out)
        # attn = (q_i * k_j * dk).sum(dim=-1)
        attn = (q * k).sum(axis=-1)
        attn = self.attn_activation(attn)
        attn = self.drop(attn)


        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = paddle.split(v_j, 3, axis=2)
        # print("x11221",v_j.shape)
        # print("vec1",vec_j.shape)
        # print("vec2",vec1.shape)
        row, col = edge_index
        aa  = vec_j[row]
        bb = vec_j[col]

        bbb = paddle.concat([aa, bb], axis=3)
        vec_j = self.lin3(bbb)
        # update scalar features
        x_ = x * attn.unsqueeze(2)
        # update vector features
        vec_ = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        # print("x111",x.shape)
        # print("vecvec",vec.shape)
        # print("dk",dk.shape)
        # vec_ = self.drop1(vec_)

        return x_, vec_     


# def get_edges_batch(n_nodes, batch_size):
#     edges = get_edges(n_nodes)
#     edge_attr = paddle.ones(len(edges[0]) * batch_size, 1)
#     edges = [paddle.LongTensor(edges[0]), paddle.LongTensor(edges[1])]
#     if batch_size == 1:
#         return edges, edge_attr
#     elif batch_size > 1:
#         rows, cols = [], []
#         for i in range(batch_size):
#             rows.append(edges[0] + n_nodes * i)
#             cols.append(edges[1] + n_nodes * i)
#         edges = [paddle.concat(rows), paddle.concat(cols)]
#     return edges, edge_attr
# class GcnEncoderGraph(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
#             pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
#         super(GcnEncoderGraph, self).__init__()
#         self.concat = concat
#         add_self = not concat
#         self.bn = bn
#         self.num_layers = num_layers
#         self.num_aggs=1

#         self.bias = True
#         if args is not None:
#             self.bias = args.bias

#         self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
#                 input_dim, hidden_dim, embedding_dim, num_layers, 
#                 add_self, normalize=True, dropout=dropout)
#         self.act = nn.ReLU()
#         self.label_dim = label_dim

#         if concat:
#             self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
#         else:
#             self.pred_input_dim = embedding_dim
#         self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
#                 label_dim, num_aggs=self.num_aggs)

#         for m in self.modules():
#             if isinstance(m, GraphConv):
#                 m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     m.bias.data = init.constant(m.bias.data, 0.0)

#     def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
#             normalize=False, dropout=0.0):
#         conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
#                 normalize_embedding=normalize, bias=self.bias)
#         conv_block = nn.ModuleList(
#                 [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
#                         normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
#                  for i in range(num_layers-2)])
#         conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
#                 normalize_embedding=normalize, bias=self.bias)
#         return conv_first, conv_block, conv_last

#     def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
#         pred_input_dim = pred_input_dim * num_aggs
#         if len(pred_hidden_dims) == 0:
#             pred_model = nn.Linear(pred_input_dim, label_dim)
#         else:
#             pred_layers = []
#             for pred_dim in pred_hidden_dims:
#                 pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
#                 pred_layers.append(self.act)
#                 pred_input_dim = pred_dim
#             pred_layers.append(nn.Linear(pred_dim, label_dim))
#             pred_model = nn.Sequential(*pred_layers)
#         return pred_model
#     def apply_bn(self, x):
#         ''' Batch normalization of 3D tensor x
#         '''
#         bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
#         return bn_module(x)

#     def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

#         ''' Perform forward prop with graph convolution.
#         Returns:
#             Embedding matrix with dimension [batch_size x num_nodes x embedding]
#         '''

#         x = conv_first(x, adj)
#         x = self.act(x)
#         if self.bn:
#             x = self.apply_bn(x)
#         x_all = [x]
#         #out_all = []
#         #out, _ = torch.max(x, dim=1)
#         #out_all.append(out)
#         for i in range(len(conv_block)):
#             x = conv_block[i](x,adj)
#             x = self.act(x)
#             if self.bn:
#                 x = self.apply_bn(x)
#             x_all.append(x)
#         x = conv_last(x,adj)
#         x_all.append(x)
#         # x_tensor: [batch_size x num_nodes x embedding]
#         x_tensor = torch.cat(x_all, dim=2)
#         if embedding_mask is not None:
#             x_tensor = x_tensor * embedding_mask
#         return x_tensor

#     def forward(self, x, adj, batch_num_nodes=None, **kwargs):
#         # mask
#         max_num_nodes = adj.size()[1]
#         if batch_num_nodes is not None:
#             self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
#         else:
#             self.embedding_mask = None

#         # conv
#         print(x.shape)
#         x = self.conv_first(x, adj)
#         x = self.act(x)
#         if self.bn:
#             x = self.apply_bn(x)
#         out_all = []
#         out, _ = torch.max(x, dim=1)
#         out_all.append(out)
#         for i in range(self.num_layers-2):
#             x = self.conv_block[i](x,adj)
#             x = self.act(x)
#             if self.bn:
#                 x = self.apply_bn(x)
#             out,_ = torch.max(x, dim=1)
#             out_all.append(out)
#             if self.num_aggs == 2:
#                 out = torch.sum(x, dim=1)
#                 out_all.append(out)
#         x = self.conv_last(x,adj)
#         #x = self.act(x)
#         out, _ = torch.max(x, dim=1)
#         out_all.append(out)
#         if self.num_aggs == 2:
#             out = torch.sum(x, dim=1)
#             out_all.append(out)
#         if self.concat:
#             output = torch.cat(out_all, dim=1)
#         else:
#             output = out
        
#         ypred = self.pred_model(output)
#         print(output.shape)
#         return ypred

#     def loss(self, pred, label, type='softmax'):
#         # softmax + CE
#         if type == 'softmax':
#             return F.cross_entropy(pred, label, reduction='mean')
#         elif type == 'margin':
#             batch_size = pred.size()[0]
#             label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
#             label_onehot.scatter_(1, label.view(-1,1), 1)
#             return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
#         #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())



# class SoftPoolingGcnEncoder(GcnEncoderGraph):
#     def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
#             assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
#             pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
#             assign_input_dim=-1, args=None):
#         '''
#         Args:
#             num_layers: number of gc layers before each pooling
#             num_nodes: number of nodes for each graph in batch
#             linkpred: flag to turn on link prediction side objective
#         '''

#         super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
#                 num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
#         add_self = not concat
#         self.num_pooling = num_pooling
#         self.linkpred = linkpred
#         self.assign_ent = True

#         # GC
#         self.conv_first_after_pool = nn.ModuleList()
#         self.conv_block_after_pool = nn.ModuleList()
#         self.conv_last_after_pool = nn.ModuleList()
#         for i in range(num_pooling):
#             # use self to register the modules in self.modules()
#             conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
#                     self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
#                     add_self, normalize=True, dropout=dropout)
#             self.conv_first_after_pool.append(conv_first2)
#             self.conv_block_after_pool.append(conv_block2)
#             self.conv_last_after_pool.append(conv_last2)

#         # assignment
#         assign_dims = []
#         if assign_num_layers == -1:
#             assign_num_layers = num_layers
#         if assign_input_dim == -1:
#             assign_input_dim = input_dim

#         self.assign_conv_first_modules = nn.ModuleList()
#         self.assign_conv_block_modules = nn.ModuleList()
#         self.assign_conv_last_modules = nn.ModuleList()
#         self.assign_pred_modules = nn.ModuleList()
#         assign_dim = int(max_num_nodes * assign_ratio)
#         for i in range(num_pooling):
#             assign_dims.append(assign_dim)
#             assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
#                     assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
#                     normalize=True)
#             assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
#             assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)


#             # next pooling layer
#             assign_input_dim = self.pred_input_dim
#             assign_dim = int(assign_dim * assign_ratio)

#             self.assign_conv_first_modules.append(assign_conv_first)
#             self.assign_conv_block_modules.append(assign_conv_block)
#             self.assign_conv_last_modules.append(assign_conv_last)
#             self.assign_pred_modules.append(assign_pred)

#         self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
#                 label_dim, num_aggs=self.num_aggs)

#         for m in self.modules():
#             if isinstance(m, GraphConv):
#                 m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
#                 if m.bias is not None:
#                     m.bias.data = init.constant(m.bias.data, 0.0)

#     def forward(self, x, adj, batch_num_nodes, **kwargs):
#         if 'assign_x' in kwargs:
#             x_a = kwargs['assign_x']
#         else:
#             x_a = x
#         print('x',x.shape)
#         # mask
#         max_num_nodes = adj.size()[1]
#         if batch_num_nodes is not None:
#             embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
#         else:
#             embedding_mask = None

#         out_all = []

#         #self.assign_tensor = self.gcn_forward(x_a, adj, 
#         #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
#         #        embedding_mask)
#         ## [batch_size x num_nodes x next_lvl_num_nodes]
#         #self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
#         #if embedding_mask is not None:
#         #    self.assign_tensor = self.assign_tensor * embedding_mask
#         # [batch_size x num_nodes x embedding_dim]
#         embedding_tensor = self.gcn_forward(x, adj,
#                 self.conv_first, self.conv_block, self.conv_last, embedding_mask)

#         out, _ = torch.max(embedding_tensor, dim=1)
#         out_all.append(out)
#         if self.num_aggs == 2:
#             out = torch.sum(embedding_tensor, dim=1)
#             out_all.append(out)

#         for i in range(self.num_pooling):
#             if batch_num_nodes is not None and i == 0:
#                 embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
#             else:
#                 embedding_mask = None

#             self.assign_tensor = self.gcn_forward(x_a, adj, 
#                     self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
#                     embedding_mask)
#             # [batch_size x num_nodes x next_lvl_num_nodes]
#             self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
#             if embedding_mask is not None:
#                 self.assign_tensor = self.assign_tensor * embedding_mask

#             # update pooled features and adj matrix
#             x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
#             adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
#             x_a = x
        
#             embedding_tensor = self.gcn_forward(x, adj, 
#                     self.conv_first_after_pool[i], self.conv_block_after_pool[i],
#                     self.conv_last_after_pool[i])


#             out, _ = torch.max(embedding_tensor, dim=1)
#             out_all.append(out)
#             if self.num_aggs == 2:
#                 #out = torch.mean(embedding_tensor, dim=1)
#                 out = torch.sum(embedding_tensor, dim=1)
#                 out_all.append(out)


#         if self.concat:
#             output = torch.cat(out_all, dim=1)
#         else:
#             output = out
#         print('ut',output.shape)
        
#         ypred = self.pred_model(output)
#         print('kkkk',ypred.shape)
#         return ypred

#     def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
#         ''' 
#         Args:
#             batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
#         '''
#         eps = 1e-7
#         loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
#         if self.linkpred:
#             max_num_nodes = adj.size()[1]
#             pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
#             tmp = pred_adj0
#             pred_adj = pred_adj0
#             for adj_pow in range(adj_hop-1):
#                 tmp = tmp @ pred_adj0
#                 pred_adj = pred_adj + tmp
#             pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
#             #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
#             #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
#             #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
#             self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
#             if batch_num_nodes is None:
#                 num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
#                 print('Warning: calculating link pred loss without masking')
#             else:
#                 num_entries = np.sum(batch_num_nodes * batch_num_nodes)
#                 embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
#                 adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
#                 self.link_loss[(1-adj_mask).bool()] = 0.0

#             self.link_loss = torch.sum(self.link_loss) / float(num_entries)
#             #print('linkloss: ', self.link_loss)
#             return loss + self.link_loss
#         return loss



class AdaptiveFourierNeuralOperator(nn.Layer):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w

        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02

        self.w1 = paddle.create_parameter([2, self.num_blocks, self.block_size, self.block_size],dtype="float32")
        # self.w1 = paddle.create_parameter([2, self.num_blocks, self.block_size, self.block_size], dtype="float32")
        # self.w1 = paddle.nn.ParameterList(self.scale * paddle.randn([2, self.num_blocks, self.block_size, self.block_size]))
        self.b1 = paddle.create_parameter([2, self.num_blocks, self.block_size],dtype="float32")
        self.w2 = paddle.create_parameter([2, self.num_blocks, self.block_size, self.block_size],dtype="float32")
        self.b2 = paddle.create_parameter([2, self.num_blocks, self.block_size],dtype="float32")
        self.relu = nn.ReLU()


        self.bias = nn.Conv1D(self.hidden_size, self.hidden_size, 1)
        self.softshrink = 0.00

    def multiply(self, input, weights):
        # print(input)
        # print(weights)
        return paddle.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, spatial_size=None):
        # print(x.shape)
        x = x.squeeze(0)
        a_sqrt = x.shape[0]**0.5
        a_sqrt_plus = math.ceil(a_sqrt)
        b = a_sqrt_plus * a_sqrt_plus
        c = b-x.shape[0]
        pad = [0,c,0,0]
        pp = nn.Pad2D(padding=pad)
        x = pp(x)
        x = x.unsqueeze(0)
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        if self.bias:
            bias = paddle.transpose(self.bias(paddle.transpose(x, perm=[0, 2, 1])),perm=[0,2,1])
        else:
            bias = paddle.zeros(x.shape, device=x.device)
        # x = paddle.tile(x,[x.shape[1],1,1])
        # print(x.shape)
        x = x.reshape([B, a, b, C])
        # print(x.shape)

        x = paddle.fft.rfft2(x, axes=(1, 2), norm='ortho')
        # print(x.shape)

        x = x.reshape([B, x.shape[1], x.shape[2],self.num_blocks, self.block_size])

        x_real_1 = F.relu(self.multiply(paddle.real(x), self.w1[0]) - self.multiply(paddle.imag(x), self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(paddle.real(x), self.w1[1]) + self.multiply(paddle.imag(x), self.w1[0]) + self.b1[1])
        x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x = paddle.stack([x_real_2, x_imag_2], axis=-1)
        # print(x.shape)

        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
        # print("jj",x.shape)

        x = paddle.as_complex(x)
        # print(x.shape)

        x = x.reshape([B, x.shape[1], x.shape[2], self.hidden_size])
        # print(x.shape)

        x = paddle.fft.irfft2(x, s=(b, a), axes=(1, 2), norm='ortho')
        x = x.reshape([B, N, C])
        x= x + bias
        x= x.squeeze(0)
        x = x[:(N-c),:]
        # x = paddle.mean(x,axis=2)
        # print(x.shape)

        return x
class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        return x
class Block(nn.Layer):
    def __init__(self, dim, mlp_ratio=4., drop=0.2, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, use_fno=False, use_blocks=False):
        super().__init__()
        # args = get_args()
        self.norm1 = norm_layer(dim)


        self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        # if self.double_skip:
        #     x = x + residual
        #     residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
    
class AFNONet(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0.2, drop_path_rate=0., norm_layer=None, 
                 dropcls=0, use_fno=False, use_blocks=False):
        super().__init__()
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)
        h = img_size // patch_size
        w = h // 2 + 1
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=0.2, drop_path=1, norm_layer=nn.LayerNorm, h=h, w=w, use_fno=use_fno, use_blocks=use_blocks)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(128)
        self.final_dropout = nn.Identity()
    def forward(self, x):
        B = x.shape[0]
        # x = self.patch_embed(x)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x=x.squeeze(0)
        # x = self.norm(x)
        # x = self.final_dropout(x)
        return x