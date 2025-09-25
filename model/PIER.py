import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from utils import DNN
from model.StateEncoder import StateEncoder
from itertools import permutations
class HashFunction(nn.Module):
    def __init__(self, embedding_dim, hash_dim):
        super(HashFunction, self).__init__()
        self.linear = nn.Linear(embedding_dim, hash_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        # 线性变换
        hash_values = self.linear(embeddings)
        # 使用 sigmoid 函数将值映射到 [0, 1] 区间
        hash_values = self.sigmoid(hash_values)
        # 将值转换为二进制（这里简单将大于 0.5 的设为 1，小于等于 0.5 的设为 0）
        binary_hash = (hash_values > 0.5).float()
        return binary_hash


def hamming_distance(hash1, hash2):
    
    return torch.sum(torch.abs(hash1 - hash2), dim=1)

class PIER(nn.Module):
    def __init__(self, args):
        super(PIER, self).__init__()
        ## FPSM
        self.his_enc = StateEncoder(args)
        self.stage1_choose = args.stage1_choose
        self.stage1_origin = 50

        ## OCPM
        self.prm_pv_input_dim = args.data_latent_dim
        self.prm_pv_hidden_dims = args.hidden_dims
        
        self.prm_encoder_n_head = args.transformer_n_head
        
        self.prm_encoder_n_layer = args.transformer_n_layer  
        self.hash_function = HashFunction(self.prm_pv_input_dim, self.prm_pv_input_dim)

        self.PVUserInputMap = nn.Linear(self.prm_pv_input_dim, self.prm_pv_input_dim)
        self.PVItemInputMap = nn.Linear(self.prm_pv_input_dim, self.prm_pv_input_dim)
        self.PVInputNorm = nn.LayerNorm(self.prm_pv_input_dim)
        self.PVOutput = DNN(self.prm_pv_input_dim, self.prm_pv_hidden_dims, self.prm_pv_hidden_dims,
                            dropout_rate=args.dropout, do_batch_norm=True)

        self.PVPosEmb = nn.Embedding(self.stage1_choose, self.prm_pv_hidden_dims)
        self.PV_pos_emb_getter = torch.arange(self.stage1_choose, dtype=torch.long).cuda()
        self.PV_attn_mask = ~torch.tril(torch.ones((self.stage1_choose, self.stage1_choose), dtype=torch.bool)).cuda()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.prm_pv_hidden_dims, 
                                                   dim_feedforward = self.prm_pv_hidden_dims, 
                                                   nhead=self.prm_encoder_n_head, dropout = args.dropout, 
                                                   batch_first = True)
        self.PRMEncoder = nn.TransformerEncoder(encoder_layer, num_layers=self.prm_encoder_n_layer)
        
        # output layer of PRM
        self.PRMOutput = nn.Linear(self.prm_pv_hidden_dims, 1)

        self.proj = nn.Linear(31*self.prm_pv_input_dim*6, 31*self.prm_pv_input_dim)

        ## OAU TAU CPU
        self.OAU_prm_pv_hidden_dims = args.hidden_dims
        OAU_encoder_layer = nn.TransformerEncoderLayer(d_model=self.OAU_prm_pv_hidden_dims,
                                                   dim_feedforward=self.OAU_prm_pv_hidden_dims,
                                                   nhead=args.transformer_n_head, dropout=args.dropout,
                                                   batch_first=True)
        self.OAU_transformer = nn.TransformerEncoder(OAU_encoder_layer, num_layers=args.transformer_n_layer)

        self.TAU_linear = nn.Linear(args.hidden_dims, args.hidden_dims)

        self.CPU_Output = nn.Linear(args.hidden_dims, 1)
        
        self.criterion = nn.BCELoss()
        self.contrastive_loss_weight = 0.1

    def forward(self, batch, candidate_score, candidate_indicies):
        labels=torch.cat((batch[4],torch.zeros(batch[4].size(0),self.stage1_choose-batch[4].size(1)).cuda()),dim=1)
        candidate_indicies=candidate_indicies.long()
        perm = torch.randperm(self.stage1_choose)

        # # 根据随机顺序重新排列 candidate_indicies 和 labels
        shuffled_candidate_indicies = candidate_indicies[:, perm]
        new_positions = [perm.tolist().index(i) for i in range(6)] 
        shuffled_labels = labels[:,perm]
        candidate_item_emb= self.his_enc.iIDEmb(shuffled_candidate_indicies)
        user_state=self.his_enc(batch)
        # batch size
        B = user_state.shape[0]        
        # input layer
        # (B, 1, pv_input_dim)
        user_input = self.PVUserInputMap(user_state).view(B,1,self.prm_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.prm_pv_input_dim))\
                            .view(B,self.stage1_choose,self.prm_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.prm_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.prm_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        # PRM transformer encoder output (B, C, enc_dim)
        PRM_encoder_output = self.PRMEncoder(pv_E, mask = self.PV_attn_mask)
        
        # PRM reranked score (B, C)
        rerank_score = self.PRMOutput((candidate_score.unsqueeze(-1)+PRM_encoder_output).view(B*self.stage1_choose,self.prm_pv_hidden_dims))\
                                    .view(B,self.stage1_choose)
        score = torch.sigmoid(rerank_score)
        # _, indices = torch.topk(rerank_prob, k = self.slate_size, dim = 1)
        x_t=self.add_noise(shuffled_labels)
        x_0 = torch.sigmoid(self.PRMOutput((x_t.unsqueeze(-1)+PRM_encoder_output).view(B*self.stage1_choose,self.prm_pv_hidden_dims))\
                                    .view(B,self.stage1_choose))         
        
        # return criterion(score, labels)
        bce_loss= self.criterion(score, shuffled_labels)+0.1*self.criterion(x_0, shuffled_labels)
        ## loss2
        distances = []

        labels_new=labels[new_positions]
        score_new=score[new_positions]
        candidate_item_6=candidate_item_emb[:,new_positions]
        # full_rank = torch.tensor(list(permutations(range(6))), dtype=torch.float)

        # 重复256次
        # full_rank = full_rank.unsqueeze(0).expand(256, -1, -1)
        # candi_=torch.mean(candidate_item_6[full_rank], dim=1, keepdim=True) 
        
        permutation_can = torch.empty(B, 31, 6,self.prm_pv_input_dim ).cuda()
        position_=torch.empty(31,6).cuda()
        # torch.cuda.manual_seed(3)    
        # 执行打乱操作
        for i in range(30):
            # 随机打乱6维度的索引
            perm_ = torch.randperm(6)
            permutation_can[:, i] = candidate_item_6[:,perm_]  # 将打乱后的数据存储
            position_[i]=perm_
        permutation_can[:, 30] = candidate_item_6 
        position_[30]=torch.arange(6)


        # permutation_can = permutation_can.mean(dim=2)  # shape: [256, 10, 64]

        candidate_hashes = self.hash_function(self.proj(permutation_can.reshape(B,31*self.prm_pv_input_dim*6)).reshape(B,31,self.prm_pv_input_dim))
        history_hashes = self.hash_function(user_input)

        for i in range(candidate_hashes.shape[1]):
            
           dist = hamming_distance(candidate_hashes[:,i,:], history_hashes[:, 0, :])
           distances.append(dist)
        distances = torch.stack(distances, dim=1)
        # 选择距离最小的前 K 个排列
        min_values, min_indices = torch.min(distances, dim=1)
        max_values, max_indices = torch.max(distances, dim=1)
        pos_rank=position_[min_indices].long()
        positive_score=score_new[pos_rank].clone().detach()
        neg_rank=position_[max_indices].long()
        negative_score=score_new[neg_rank].clone().detach()



        margin = 0.5
        contrastive_loss = torch.mean(torch.clamp(margin - positive_score + negative_score, min=0))
        
        

        total_loss = bce_loss+0.1*contrastive_loss
        return total_loss

    def forward_eval(self, batch, candidate_score, candidate_indicies):
        
        candidate_indicies=candidate_indicies.long()
        candidate_item_emb= self.his_enc.iIDEmb(candidate_indicies)
        user_state=self.his_enc(batch)
        # batch size
        B = user_state.shape[0]        

        user_input = self.PVUserInputMap(user_state).view(B,1,self.prm_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.prm_pv_input_dim))\
                            .view(B,self.stage1_choose,self.prm_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.prm_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.prm_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        PRM_encoder_output = self.PRMEncoder(pv_E, mask = self.PV_attn_mask)
        
        # PRM reranked score (B, C)
        rerank_score = self.PRMOutput(PRM_encoder_output.view(B*self.stage1_choose,self.prm_pv_hidden_dims))\
                                    .view(B,self.stage1_choose)
        score = torch.sigmoid(rerank_score)

#         oau_output = self.OAU_transformer(pv_E, mask=self.PV_attn_mask)
#         # 假设历史排列和目标排列相同
#         tau_output = self.TAU_linear(oau_output) * oau_output
#         rerank_score = self.CPU_Output(tau_output.view(B * self.stage1_choose, self.prm_pv_hidden_dims)).view(B, self.stage1_choose)
#         score = torch.sigmoid(rerank_score)
        
        
        return score



    def add_noise(self,labels):
            # noise = th.randn_like(labels)
        alpha = 0.5
        beta = 0.5
        beta_dist = torch.distributions.Beta(alpha, beta)

        # 采样β分布噪声
        noise =(beta_dist.sample(labels.shape) - 0.5).cuda()  # 将噪声范围调整到 [-0.5, 0.5]
        x_t=labels+0.1*noise
        x_t=(x_t - x_t.min(dim=1, keepdim=True)[0]) / \
                (x_t.max(dim=1, keepdim=True)[0] - x_t.min(dim=1, keepdim=True)[0])
        return x_t

    # def add_noise(self, candidate_state, user_state,labels):
    #     # alpha = 0.5
    #     # beta = 0.5
    #     # beta_dist = torch.distributions.Beta(alpha, beta)
    #     # # 采样β分布噪声
    #     # noise = (beta_dist.sample(labels.shape) - 0.5).cuda()  # 将噪声范围调整到 [-0.5, 0.5]
    #     x_t=labels+0.1*F.normalize(self.bn1(self.fc1(candidate_state+user_state.unsqueeze(1).repeat(1,self.stage1_choose,1))).squeeze(),p=2,dim=1)
    #     # x_t=labels+0.09*F.normalize(self.bn1(self.fc1(candidate_state+user_state.unsqueeze(1).repeat(1,self.stage1_choose,1))).squeeze(),p=2,dim=1)+0.01*noise
    #     x_t=(x_t - x_t.min(dim=1, keepdim=True)[0]) / \
    #             (x_t.max(dim=1, keepdim=True)[0] - x_t.min(dim=1, keepdim=True)[0])
    #     noise=F.normalize(self.bn1(self.fc1(candidate_state+user_state.unsqueeze(1).repeat(1,self.stage1_choose,1))).squeeze(),p=2,dim=1)
    #     return x_t

