import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import *

class GeomMPNN_2D(nn.Layer):
    def __init__(self, node_in_dim, edge_in_dim, rbf_dim, hidden_dim, max_dist, \
                       num_conv, num_pool, dropout, dropout_pool, activation=F.relu):
        super(GeomMPNN_2D, self).__init__()
        self.num_conv = num_conv
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.geometry_embed = GeometryEmbedding(rbf_dim, hidden_dim, max_dist, activation)
        self.node2edge_layers = nn.LayerList()
        self.edge2edge_layers = nn.LayerList()
        self.edge2node_layers = nn.LayerList()
        self.attn_pool_layers = nn.LayerList()
        for i in range(num_conv):
            node_in_dim = node_in_dim if i == 0 else hidden_dim
            self.node2edge_layers.append(EdgeAggConv(node_in_dim, edge_in_dim, hidden_dim, activation))
            self.edge2edge_layers.append(Angle2DConv(hidden_dim, hidden_dim, activation, dropout))
            self.edge2node_layers.append(Dist2DConv(hidden_dim, node_in_dim, hidden_dim, activation, dropout))
        for _ in range(num_pool):
            self.attn_pool_layers.append(AttentivePooling(hidden_dim, dropout=dropout_pool))
        # self.output_layer = OutputLayer(node_in_dim, mlp_dims, num_pool, dropout_pool)
    
    def forward(self, a2a_graph, e2a_graph, e2e_graph):
        node_feat = paddle.cast(a2a_graph.node_feat['feat'], 'float32')
        bond_feat = paddle.cast(e2e_graph.node_feat['feat'], 'float32')
        dist_feat = paddle.cast(a2a_graph.edge_feat['dist'], 'float32')
        angle_feat = paddle.cast(e2e_graph.edge_feat['angle'].reshape([-1,1]), 'float32')
        # print(a2a_graph.num_edges, a2a_graph.edge_feat['dist'].shape)
        
        dist_h, angle_h = self.geometry_embed([dist_feat], [angle_feat])
        dist_h, angle_h = dist_h[0], angle_h[0]
        for i in range(self.num_conv):
            edge_feat = self.node2edge_layers[i](a2a_graph, node_feat, bond_feat)
            edge_feat = self.edge2edge_layers[i](e2e_graph, edge_feat, angle_h)
            node_feat = self.edge2node_layers[i](e2a_graph, edge_feat, node_feat, dist_h)
        
        graph_feat = self.pool(a2a_graph, node_feat)
        for attn_pool in self.attn_pool_layers:
            graph_feat = attn_pool(a2a_graph, node_feat, graph_feat)
        return graph_feat


class GeomMPNN_3D(nn.Layer):
    def __init__(self, node_in_dim, rbf_dim, hidden_dim, cut_dist, \
                       num_conv, num_pool, num_dist, num_angle, dropout, dropout_pool, activation=F.relu):
        super(GeomMPNN_3D, self).__init__()
        self.num_conv = num_conv
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.geometry_embed = GeometryEmbedding(rbf_dim, hidden_dim, cut_dist, activation)
        self.node2edge_layers = nn.LayerList()
        self.a = nn.Linear(256,128)
        self.b = nn.Linear(57,128)
        self.bkk = nn.Linear(57,128)
        self.edge2edge_layers = nn.LayerList()
        self.Eqt = nn.LayerList()
        self.edge2node_layers = nn.LayerList()
        self.attn_pool_layers = nn.LayerList()
        for i in range(num_conv):
            node_in_dim = node_in_dim if i == 0 else hidden_dim * num_dist
            self.node2edge_layers.append(EdgeAggConv(node_in_dim, hidden_dim, hidden_dim, activation))
            self.edge2edge_layers.append(Angle3DConv(hidden_dim, hidden_dim, num_angle, activation, dropout))
            self.edge2node_layers.append(Dist3DConv(hidden_dim * num_angle, node_in_dim, hidden_dim, num_dist, activation, dropout))
        for _ in range(num_pool):
            self.attn_pool_layers.append(AttentivePooling(hidden_dim, dropout=0.1))
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
        self.EGNN = EGNN(in_node_nf=57, hidden_nf=128,out_node_nf=128,in_edge_nf=128,device='cuda:0',act_fn=nn.Silu(),n_layers=1,residual=True, attention=False, normalize=False, tanh=False)
        self.EGNN1 = EGNN(in_node_nf=128, hidden_nf=128,out_node_nf=128,in_edge_nf=128,device='cuda:0',act_fn=nn.Silu(),n_layers=1,residual=True, attention=False, normalize=False, tanh=False)
        self.out_norm = nn.LayerNorm(128)




    def forward(self, a2a_graph, e2a_graph_list, e2e_graph_list):
        node_feat = paddle.cast(a2a_graph.node_feat['feat'], 'float32')
        x = paddle.cast(a2a_graph.node_feat['coord'], 'float32')
        dist_feat = paddle.cast(a2a_graph.edge_feat['dist'], 'float32')
        # dist_feat_list = [paddle.cast(e2a_g.edge_feat['dist'], 'float32') for e2a_g in e2a_graph_list]
        # angle_feat_list = [paddle.cast(e2e_g.edge_feat['angle'], 'float32').reshape([-1,1]) for e2e_g in e2e_graph_list]

        # dist_h_list, angle_h_list = self.geometry_embed(dist_feat_list + [dist_feat], angle_feat_list)
        # dist_h, dist_h_list = dist_h_list[-1], dist_h_list[:-1]
        # for i in range(self.num_conv):
        #     edge_feat = self.node2edge_layers[i](a2a_graph, node_feat, dist_h)
        #     edge_feat = self.edge2edge_layers[i](e2e_graph_list, edge_feat, angle_h_list)
        #     node_feat = self.edge2node_layers[i](e2a_graph_list, edge_feat, node_feat, dist_h_list)
        # print(node_feat.shape)
        alledge = (e2a_graph_list.edges).T
        edge1 = (a2a_graph.edges).T

        atom_feat = node_feat
        # print(a2a_g.num_edges, a2a_g.edge_feat['dist'].shape)
        # x = a2a_graph.node_feat['coord']
        atom_h = self.bkk(atom_feat)
        ha = atom_h
        # dist_h = self.input_layer(dist_feat)

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

        # atom_h = self.b(node_feat)
        # atom_h2 = atom_h
        # edge1 = (a2a_graph.edges).T
        # edge_weight, edge_vec = self.dis(edge1,x)
        # edge_attr = self.dist_rbf(edge_weight)
        # mask = edge1[0] != edge1[1]
        # edge_vec[mask] = edge_vec[mask] / paddle.norm(edge_vec[mask], axis=1).unsqueeze(1)
        # vec = paddle.zeros([atom_h.shape[0], 3, atom_h.shape[1]])
        # for attn in self.Eqt:
        #     dx, dvec = attn(atom_h, vec, edge1, edge_weight, edge_attr, edge_vec)
        #     atom_h = atom_h + dx
        #     vec = vec + dvec
        # # print(node_feat.shape)
        # atom_h = self.out_norm(atom_h)
        
        # h, x = self.EGNN(node_feat, x, edge1)
        # hh=h
        # h, x = self.EGNN1(h,x,edge1)
        h = paddle.concat([ha,vvvv],axis=1)
        nnn= self.a(h)

        graph_feat = self.pool(a2a_graph, nnn)
        # for attn_pool in self.attn_pool_layers:
        #     graph_feat = attn_pool(a2a_graph, nnn, graph_feat)
        return graph_feat


class GeomMPNN(nn.Layer):
    def __init__(self, node_in_dim, edge_in_dim, atom_in_dim, rbf_dim, hidden_dim, max_dist_2d, cut_dist_3d, mlp_dims, spa_weight, \
                       num_conv, num_pool, num_dist, num_angle, dropout, dropout_pool, task_dim, activation=F.relu):
        super(GeomMPNN, self).__init__()
        self.spa_weight = spa_weight
        self.mpnn_2d = GeomMPNN_2D(node_in_dim, edge_in_dim, rbf_dim, hidden_dim, max_dist_2d, \
                                   num_conv, num_pool, dropout=0, dropout_pool=dropout_pool, activation=F.relu)
        self.mpnn_3d = GeomMPNN_3D(atom_in_dim, rbf_dim, hidden_dim, cut_dist_3d,  num_conv, \
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
        # for layer in self.mpnn_3d.edge2edge_layers:
        #     w_g = paddle.stack([conv.G.weight for conv in layer.conv_layer])
        #     loss += self.spa_weight * paddle.sum((w_g[1:, :, :] - w_g[:-1, :, :])**2)
        return loss
    
    def loss_cls(self, y_hat, y):
        y_mask = paddle.where(y == -1, paddle.to_tensor([0.]), paddle.ones_like(y))
        y_cal = paddle.where(y == -1, paddle.to_tensor([0.]), y)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_cal, reduction='none')
        loss = loss.sum() / y_mask.sum()
        for layer in self.mpnn_3d.edge2edge_layers:
            w_g = paddle.stack([conv.G.weight for conv in layer.conv_layer])
            loss += self.spa_weight * paddle.sum((w_g[1:, :, :] - w_g[:-1, :, :])**2)
        return loss

    def forward(self, graph_2d, graph_3d):
        # feat_2d = self.mpnn_2d(*graph_2d)
        feat_3d = self.mpnn_3d(*graph_3d)
        # feat_2d = self.trans_2d_layer(feat_2d)
        # print(feat_3d.shape)
        feat_3d = self.trans_2d_layer(feat_3d)
        # graph_feat = feat_2d + feat_3d
        graph_feat = feat_3d

        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output


class GeomGCL(nn.Layer):
    def __init__(self, node_in_dim, edge_in_dim, atom_in_dim, rbf_dim, hidden_dim, max_dist_2d, cut_dist_3d, spa_weight, gcl_weight, gcl_tau, \
                       num_conv, num_pool, num_dist, num_angle, dropout, dropout_pool, activation=F.relu):
        super(GeomGCL, self).__init__()
        self.spa_weight = spa_weight
        self.gcl_weight = gcl_weight
        self.gcl_tau = gcl_tau
        self.mpnn_2d = GeomMPNN_2D(node_in_dim, edge_in_dim, rbf_dim, hidden_dim, max_dist_2d, \
                                   num_conv, num_pool, dropout, dropout_pool, activation)
        self.mpnn_3d = GeomMPNN_3D(atom_in_dim, rbf_dim, hidden_dim, cut_dist_3d,  num_conv, \
                                   num_pool, num_dist, num_angle, dropout, dropout_pool, activation)
    
        self.trans_2d_layer = nn.Linear(hidden_dim, hidden_dim)
        self.trans_3d_layer = nn.Linear(hidden_dim * num_dist, hidden_dim)
        self.proj_2d_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_2d_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_3d_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_3d_fc2 = nn.Linear(hidden_dim, hidden_dim)


    def projection_2d(self, z):
        z = F.elu(self.proj_2d_fc1(z))
        return self.proj_2d_fc2(z)

    def projection_3d(self, z):
        z = F.elu(self.proj_3d_fc1(z))
        return self.proj_3d_fc2(z)
    
    def sim(self, zi, zj):
        # zi = F.normalize(zi)
        # zj = F.normalize(zj)
        zi = zi/paddle.sqrt((zi*zi).sum(1)).unsqueeze(1)
        zj = zj/paddle.sqrt((zj*zj).sum(1)).unsqueeze(1)
        return paddle.mm(zi, zj.t())

    def gcl_loss(self, z_2d, z_3d):
        z_2d = self.projection_2d(z_2d)
        z_3d = self.projection_3d(z_3d)
        f = lambda x: paddle.exp(x / self.gcl_tau)
        between_sim = f(self.sim(z_2d, z_3d))
        return -paddle.log(paddle.diag(between_sim) / between_sim.sum(1))-paddle.log(paddle.diag(between_sim) / between_sim.sum(0))

    def loss(self, z_2d, z_3d):
        loss = self.gcl_weight * self.gcl_loss(z_2d, z_3d).sum()
        for layer in self.mpnn_3d.edge2edge_layers:
            w_g = paddle.stack([conv.G.weight for conv in layer.conv_layer])
            loss += self.spa_weight * paddle.sum((w_g[1:, :, :] - w_g[:-1, :, :])**2)
        return loss

    def forward(self, graph_2d, graph_3d):
        feat_2d = self.mpnn_2d(*graph_2d)
        feat_3d = self.mpnn_3d(*graph_3d)
        feat_2d = self.trans_2d_layer(feat_2d)
        feat_3d = self.trans_3d_layer(feat_3d)
        return feat_2d, feat_3d