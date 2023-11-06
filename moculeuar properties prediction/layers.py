import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

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

        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out   

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index


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

        radial = paddle.sum(coord_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = paddle.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
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



   
    def forward(self, h, x, edge):
        h = self.embedding_in(h)

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

        dist = dist.unsqueeze(-1)


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
        self.ww = nn.Linear(128, 128)
        self.qq = nn.Linear(128, 128)
        self.uu = nn.Linear(128, 128)
        self.kk = nn.Linear(128, 128)
        self.oo = nn.Linear(131, 128)
        self.cc = nn.Linear(1, 1)
        self.tt = nn.Linear(128, 3)
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

        return radial ,coord_diff
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index


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

        agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])

        coord = coord + agg
        return coord
    def forward(self, x, vec, edge_index, c_ij, f_ij, d_ij,coord,ha,alledge):

        h = self.hhh(ha)
        row, col = edge_index
        r, c = alledge
        edge_attr = None
        node_attr = None
        radial,coord_diff = self.coord2radial(edge_index,coord)
        M_ij = self.edge_model(h[row], h[col], radial,edge_attr)
        h = self.node_model(h, edge_index, M_ij,node_attr)
        coord1 = self.coord_model(coord, edge_index, coord_diff, M_ij)
        h = self.ggg(h)
        x = self.layernorm(x)
        bij = self.qq(x[r])*self.kk(x[c])
        ccij = self.cc(paddle.norm((coord[r]-coord[c]),p=2))
        aa = self.sss(bij+ccij)
        U = unsorted_segment_sum(aa*self.uu(x[r]), row, num_segments=x.shape[0])
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



        x_, vec_ = self.message1(
            edge_index,
            q,
            k,
            v,
            vec,
            coord,
            dv,
            d_ij
            )
        x_ = x_.reshape([-1, self.hidden_channels])
        vec_ = vec_.reshape([-1, 3, self.hidden_channels])
        x = unsorted_segment_sum(x_, row, num_segments=x.shape[0])
        v1,v2,v3 = paddle.split(vec_,3,axis = 1)
        v1 = unsorted_segment_sum(v1.squeeze(1), row, num_segments=vec.shape[0])
        v2 = unsorted_segment_sum(v2.squeeze(1), row, num_segments=vec.shape[0])
        v3 = unsorted_segment_sum(v3.squeeze(1), row, num_segments=vec.shape[0])
        vec = paddle.stack([v1,v2,v3],axis = 1)


        o1, o2, o3 = paddle.split(self.o_proj(x), 3, axis=1)
        dx = vec_dot * o2 + o3*U
        dvec = vec3 * o1.unsqueeze(1) + vec
        # dx = self.aaa(dx)
        dh = vec_dot*o2+h
        return dx, dvec,dh,coord1


    def message1(self, edge_index,q,k,v_j, vec_j, coord, dv, d_ij):
        attn = (q * k).sum(axis=-1)
        attn = self.attn_activation(attn)
        attn = self.drop(attn)
        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = paddle.split(v_j, 3, axis=2)
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
        return x_, vec_    