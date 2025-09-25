import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from utils import DNN
from model.StateEncoder import StateEncoder
criterion = nn.BCELoss() 

class Evaluator(nn.Module):
    '''
    Pointwise model
    '''
    
        
    def __init__(self, args):
        # TwoStageOnlinePolicy initialization: 
        # - initial_list_size, stage1_n_neg, stage1_state2z_hidden_dims, stage1_pos_offset, stage1_neg_offset, initial_loss_coef
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args): userEncoder, enc_dim, state_dim, action_dim
        super(Evaluator, self).__init__()
        self.his_enc=StateEncoder(args)
        self.e_pv_input_dim = args.data_latent_dim
        self.e_pv_hidden_dims = args.hidden_dims
    
        self.e_encoder_n_head = int(args.transformer_n_head/2)
        self.e_encoder_n_layer = args.transformer_n_layer   
        self.stage1_choose=args.stage1_choose
     
        self.PVUserInputMap = nn.Linear(self.e_pv_input_dim, self.e_pv_input_dim)
        self.PVItemInputMap = nn.Linear(self.e_pv_input_dim, self.e_pv_input_dim)
        self.PVInputNorm = nn.LayerNorm(self.e_pv_input_dim)
        self.PVOutput = DNN(self.e_pv_input_dim, self.e_pv_hidden_dims, self.e_pv_hidden_dims ,
                            dropout_rate = args.dropout, do_batch_norm = True)
        # label prediction model
        self.PVPred = nn.Linear(self.e_pv_hidden_dims, 1)
        # positional embedding
        self.PVPosEmb = nn.Embedding(self.stage1_choose, self.e_pv_hidden_dims)
        self.PV_pos_emb_getter = torch.arange(self.stage1_choose, dtype = torch.long).cuda()
        self.PV_attn_mask = ~torch.tril(torch.ones((self.stage1_choose,self.stage1_choose), dtype=torch.bool)).cuda()
        
        # encoding layer of e (transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.e_pv_hidden_dims, 
                                                   dim_feedforward = self.e_pv_hidden_dims, 
                                                   nhead=self.e_encoder_n_head, dropout = args.dropout, 
                                                   batch_first = True)
        self.eEncoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_encoder_n_layer)
        
        # output layer of e
        self.eOutput = nn.Linear(self.e_pv_hidden_dims, 1)
    
    def forward(self, batch,candidate_score,candidate_indicies):
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
        user_input = self.PVUserInputMap(user_state).view(B,1,self.e_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.e_pv_input_dim))\
                            .view(B,self.stage1_choose,self.e_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.e_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.e_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        # e transformer encoder output (B, C, enc_dim)
        e_encoder_output = self.eEncoder(pv_E, mask = self.PV_attn_mask)
        
        # e reranked score (B, C)
        rerank_score = self.eOutput(e_encoder_output.view(B*self.stage1_choose,self.e_pv_hidden_dims))\
                                    .view(B,self.stage1_choose)
        score = torch.sigmoid(rerank_score)
        # _, indices = torch.topk(rerank_prob, k = self.slate_size, dim = 1)
                
        
        # return criterion(score, labels)
        return criterion(score, shuffled_labels)
               
               
               

    def forward_eval(self, batch,candidate_score,candidate_indicies):
        candidate_indicies=candidate_indicies.long()
        candidate_item_emb= self.his_enc.iIDEmb(candidate_indicies)
        user_state=self.his_enc(batch)
        # batch size
        B = user_state.shape[0]        
        # input layer
        # (B, 1, pv_input_dim)
        user_input = self.PVUserInputMap(user_state).view(B,1,self.e_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.e_pv_input_dim))\
                            .view(B,self.stage1_choose,self.e_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.e_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.e_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        # e transformer encoder output (B, C, enc_dim)
        e_encoder_output = self.eEncoder(pv_E, mask = self.PV_attn_mask)
        
        # e reranked score (B, C)
        rerank_score = self.eOutput(e_encoder_output.view(B*self.stage1_choose,self.e_pv_hidden_dims))\
                                    .view(B,self.stage1_choose)
        score = torch.sigmoid(rerank_score)
        # _, indices = torch.topk(rerank_prob, k = self.slate_size, dim = 1)
                
        
        return score