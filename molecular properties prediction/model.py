import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import *
class EMAIL(nn.Layer):
    def __init__(self, node_in_dim, rbf_dim, hidden_dim, cut_dist, \
                       num_conv, num_pool, num_dist, num_angle, dropout, dropout_pool, activation=F.relu):
        super(EMAIL, self).__init__()
        self.num_conv = num_conv
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.a = nn.Linear(256,128)
        self.b = nn.Linear(57,128)
        self.bkk = nn.Linear(57,128)

        self.Eqt = nn.LayerList()


        # self.output_layer = OutputLayer(node_in_dim, mlp_dims, num_pool, dropout_pool)
        self.dist_rbf =ExpNormalSmearing (
            0.0, 5.0, 50, True
        )
        self.dis = Distance(
        0.0,
        5.0,
        max_num_neighbors=100,
        return_vecs=True,
        loop=True,
        )
        for i in range(3):

            self.Eqt.append(EquivariantMultiHeadAttention(
                    hidden_channels=128,
                    num_rbf=50,
                    distance_influence="both",
                    num_heads=8,
                    activation=nn.Silu(),
                    attn_activation=nn.Silu(),
                    cutoff_lower=0.0,
                    cutoff_upper=5.0,
                ))        
        self.out_norm = nn.LayerNorm(128)

    def forward(self, a2a_graph, e2a_graph_list, _):
        node_feat = paddle.cast(a2a_graph.node_feat['feat'], 'float32')
        x = paddle.cast(a2a_graph.node_feat['coord'], 'float32')
        dist_feat = paddle.cast(a2a_graph.edge_feat['dist'], 'float32')
        alledge = (e2a_graph_list.edges).T
        edge1 = (a2a_graph.edges).T
        atom_feat = node_feat
        atom_h = self.bkk(atom_feat)
        ha = atom_h
        edge_weight, edge_vec = self.dis(edge1,x)
        edge_attr = self.dist_rbf(dist_feat)
        mask = edge1[0] != edge1[1]
        edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1).unsqueeze(1)
        vec = paddle.zeros([x.shape[0], 3, atom_h.shape[1]])
        for attn in self.Eqt:
            dx, dvec,ha,x = attn(atom_h, vec, edge1, edge_weight, edge_attr, edge_vec,x,ha,alledge)
            atom_h = atom_h + dx
            vec = vec + dvec
            # h1 = ha
        vvvv = self.out_norm(atom_h)
        h = paddle.concat([ha,vvvv],axis=1)
        nnn= self.a(h)
        graph_feat = self.pool(a2a_graph, nnn)
        return graph_feat


class EMAILmodel(nn.Layer):
    def __init__(self, node_in_dim, edge_in_dim, atom_in_dim, rbf_dim, hidden_dim, max_dist_2d, cut_dist_3d, mlp_dims, spa_weight, \
                       num_conv, num_pool, num_dist, num_angle, dropout, dropout_pool, task_dim, activation=F.relu):
        super(EMAILmodel, self).__init__()
        self.spa_weight = spa_weight
        self.mpnn_3d = EMAIL(atom_in_dim, rbf_dim, hidden_dim, cut_dist_3d,  num_conv, \
                                   num_pool, num_dist, num_angle, dropout, dropout_pool, activation=F.relu)
    
        self.trans_2d_layer = nn.Linear(hidden_dim, hidden_dim)
        self.trans_3d_layer = nn.Linear(hidden_dim * num_dist, hidden_dim)

        in_dim = hidden_dim
        self.mlp = nn.LayerList()
        for out_dim in mlp_dims:
            self.mlp.append(DenseLayer(in_dim, out_dim, activation=F.relu, bias=True))
            in_dim = out_dim
        self.output_layer = nn.Linear(in_dim, task_dim, bias_attr=True)

    def loss_reg(self, y_hat, y):
        loss = F.l1_loss(y_hat, y, reduction='sum')
        return loss

    def forward(self, graph_3d):

        feat_3d = self.mpnn_3d(*graph_3d)

        feat_3d = self.trans_2d_layer(feat_3d)
        graph_feat = feat_3d

        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output
