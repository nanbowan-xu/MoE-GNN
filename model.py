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
"""
Model code for Structure-aware Interactive Graph Neural Networks (SIGN).
"""
from functools import partial

from xml.etree.ElementPath import xpath_tokenizer_re
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import SpatialInputLayer,ExpNormalSmearing,Distance,EquivariantMultiHeadAttention,AFNONet, Atom2BondLayer, Bond2BondLayer, Bond2AtomLayer, PiPoolLayer, OutputLayer,Atom2AtomLayer,TransformerConv,EGNN

class SIGN(nn.Layer):
    def __init__(self, args):
        super(SIGN, self).__init__()
        num_convs = args.num_convs
        dense_dims = args.dense_dims
        infeat_dim = args.infeat_dim
        hidden_dim = args.hidden_dim
        self.num_convs = num_convs

        cut_dist = args.cut_dist
        num_angle = args.num_angle
        merge_b2b = args.merge_b2b
        merge_b2a = args.merge_b2a

        activation = args.activation
        num_heads = args.num_heads
        feat_drop = args.feat_drop
        self.out_norm = nn.LayerNorm(128)
        self.lin2 = nn.Linear(256+128,128)
        self.lin1 = nn.Linear(36,128) 
        self.dist_rbf =ExpNormalSmearing (
            0.0, 5.0, 50, True
        )
        self.dis = Distance(
        0.0,
        5.0,
        max_num_neighbors=50,
        return_vecs=True,
        loop=True,
        )
        self.afn = AFNONet(
        img_size=128, 
        patch_size=1, embed_dim=36, depth=1, mlp_ratio=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
        self.input_layer = SpatialInputLayer(hidden_dim, cut_dist, activation=F.relu)
        self.atom2bond_layers = nn.LayerList()
        self.bond2bond_layers = nn.LayerList()
        self.bond2atom_layers = nn.LayerList()
        self.atom2atom_layers = nn.LayerList()
        self.TransformerConv = nn.LayerList()
        self.Eqt= nn.LayerList()
        for i in range(2):

            self.Eqt.append(EquivariantMultiHeadAttention(
                    hidden_channels=128,
                    num_rbf=50,
                    distance_influence="both",
                    num_heads=4,
                    activation=nn.Silu(),
                    attn_activation=nn.Silu(),
                    cutoff_lower=0.0,
                    cutoff_upper=5.0,
                ))


        self.lin = nn.Linear(512,128) 
        self.gcn = pgl.nn.conv.GATConv(36, 128, 0.2, 0.2, 4, concat=True, activation=nn.Silu())
        self.gcn1 = pgl.nn.conv.GATConv(512, 128, 0.2, 0.2, 4, concat=True, activation=nn.Silu())
        self.pool = pgl.nn.GraphPool(pool_type='sum')

        self.EGNN = EGNN(in_node_nf=36, hidden_nf=128,out_node_nf=128,in_edge_nf=128,device='cuda:0',act_fn=nn.Silu(),n_layers=1,residual=True, attention=False, normalize=False, tanh=False)
        self.EGNN1 = EGNN(in_node_nf=128, hidden_nf=128,out_node_nf=128,in_edge_nf=128,device='cuda:0',act_fn=nn.Silu(),n_layers=1,residual=True, attention=False, normalize=False, tanh=False)
        # self.EGNN2 = EGNN2(in_node_nf=128, hidden_nf=128,out_node_nf=128,in_edge_nf=128,device='cuda:0',act_fn=nn.Silu(),n_layers=1,residual=True, attention=False, normalize=False, tanh=False)
        # self.Soft = SoftPoolingGcnEncoder(max_num_nodes=200, 
        #             input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #             args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #             bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        #             assign_input_dim=assign_input_dim)
        for i in range(num_convs):
            if i == 0:
                atom_dim = infeat_dim
                
            else:
                atom_dim = hidden_dim * num_heads if 'cat' in merge_b2a else hidden_dim
                
            bond_dim = hidden_dim * num_angle if 'cat' in merge_b2b else hidden_dim


            # self.aaa = nn.Linear(512,atom_dim)
            self.atom2bond_layers.append(Atom2BondLayer(atom_dim, bond_dim=hidden_dim, activation=activation))
           
            self.atom2bond_layers.append(Atom2BondLayer(atom_dim, bond_dim=hidden_dim, activation=activation))
            self.bond2bond_layers.append(Bond2BondLayer(hidden_dim, hidden_dim, num_angle, feat_drop, merge=merge_b2b, activation=None))
            self.bond2atom_layers.append(Bond2AtomLayer(bond_dim, atom_dim, hidden_dim, num_heads, feat_drop, merge=merge_b2a, activation=activation))# bond_dim -> hidden_dim 
            self.atom2atom_layers.append(Atom2AtomLayer(hidden_dim, feat_drop, activation=activation))
            self.TransformerConv.append(TransformerConv(atom_dim,
                 128,
                 num_heads=4,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation=activation))
            
        # self.aaa = nn.Linear(512,atom_dim)  
        # print(self.atom2atom_layers)
        self.pipool_layer = PiPoolLayer(hidden_dim, hidden_dim, num_angle)
        self.output_layer = OutputLayer(hidden_dim, dense_dims)

        self.nnn = nn.Linear(3,128)
        self.softmax = nn.Softmax(axis=-1)
        self.relu = nn.ReLU()





    






    def forward(self, a2a_g, b2a_g, b2b_gl, xxx,bond_types, type_count,adj):
        atom_feat = a2a_g.node_feat['feat']
        dist_feat = a2a_g.edge_feat['dist']
        # print('ggg',a2a_g.num_nodes)
        # print((a2a_g.indegree().shape))
        edge1 = (a2a_g.edges).T
        alledge = (b2a_g.edges).T
        
        atom_feat = paddle.cast(atom_feat, 'float32')
        dist_feat = paddle.cast(dist_feat, 'float32')
        # print(a2a_g.num_edges, a2a_g.edge_feat['dist'].shape)
        x = a2a_g.node_feat['coord']
        atom_h = self.lin1(atom_feat)
        ha = atom_h
        dist_h = self.input_layer(dist_feat)

        edge_weight, edge_vec = self.dis(edge1,x)
        edge_attr = self.dist_rbf(dist_feat)
        mask = edge1[0] != edge1[1]
        edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1).unsqueeze(1)
        vec = paddle.zeros([x.shape[0], 3, atom_h.shape[1]])
        dd=[]
        for attn in self.Eqt:
            dx, dvec,ha,x = attn(atom_h, vec, edge1, edge_weight, edge_attr, edge_vec,x,ha,alledge)
            atom_h = atom_h + dx
            vec = vec + dvec
            dd.append(ha)
            # ha = u
        vvvv = self.out_norm(atom_h)
        h = paddle.concat([dd[0],dd[1],vvvv],axis=1)
        h= self.lin2(h)


        pred_inter_mat = self.pipool_layer(bond_types, type_count, dist_h)
        pred_socre = self.output_layer(a2a_g, h)
        return pred_inter_mat, pred_socre,pred_socre




        
