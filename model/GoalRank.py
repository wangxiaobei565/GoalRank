import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from utils import DNN
from model.StateEncoder import StateEncoder
from model.Evaluator import Evaluator
import math

criterion = nn.BCELoss() 




class GoalRank(nn.Module):
    '''
    Pointwise model
    '''
    
        
    def __init__(self, args):
        # TwoStageOnlinePolicy initialization: 
        # - initial_list_size, stage1_n_neg, stage1_state2z_hidden_dims, stage1_pos_offset, stage1_neg_offset, initial_loss_coef
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args): userEncoder, enc_dim, state_dim, action_dim
        super(GoalRank, self).__init__()
        self.his_enc=StateEncoder(args)
        self.lrm_pv_input_dim = args.data_latent_dim
        self.lrm_pv_hidden_dims = args.hidden_dims
        self.G_weight = args.G_weight
        self.R_guide_G_weight = args.R_guide_G_weight
        self.generative_list_num = args.generative_list_num
    
        self.lrm_encoder_n_head = int(args.transformer_n_head/2)
        self.lrm_encoder_n_layer = args.transformer_n_layer   
        self.stage1_choose=args.stage1_choose
     
        self.PVUserInputMap = nn.Linear(self.lrm_pv_input_dim, self.lrm_pv_input_dim)
        self.PVItemInputMap = nn.Linear(self.lrm_pv_input_dim, self.lrm_pv_input_dim)
        self.PVInputNorm = nn.LayerNorm(self.lrm_pv_input_dim)
        self.PVOutput = DNN(self.lrm_pv_input_dim, self.lrm_pv_hidden_dims, self.lrm_pv_hidden_dims ,
                            dropout_rate = args.dropout, do_batch_norm = True)
        # label prediction model
        self.PVPred = nn.Linear(self.lrm_pv_hidden_dims, 1)
        # positional embedding
        self.PVPosEmb = nn.Embedding(self.stage1_choose, self.lrm_pv_hidden_dims)
        self.PV_pos_emb_getter = torch.arange(self.stage1_choose, dtype = torch.long).cuda()
        self.PV_attn_mask = ~torch.tril(torch.ones((self.stage1_choose,self.stage1_choose), dtype=torch.bool)).cuda()
        
        # encoding layer of lrm (transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.lrm_pv_hidden_dims, 
                                                   dim_feedforward = self.lrm_pv_hidden_dims, 
                                                   nhead=self.lrm_encoder_n_head, dropout = args.dropout, 
                                                   batch_first = True)
        self.lrmEncoder = nn.TransformerEncoder(encoder_layer, num_layers=self.lrm_encoder_n_layer)
        
        # output layer of lrm
        self.lrmOutput = nn.Linear(self.lrm_pv_hidden_dims, 1)
        
   
        
        self.evaluator = Evaluator(args)
        evaluator_state_dict = torch.load(args.path_evaluator)
        self.evaluator.load_state_dict(evaluator_state_dict)
        self.evaluator.eval()
        self.evaluator.to(args.device)
        
    
    
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
        user_input = self.PVUserInputMap(user_state).view(B,1,self.lrm_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.lrm_pv_input_dim))\
                            .view(B,self.stage1_choose,self.lrm_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.lrm_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.lrm_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        # lrm transformer encoder output (B, C, enc_dim)
        item_input_base = self.PVItemInputMap(pv_E.view(B*self.stage1_choose,self.lrm_pv_input_dim))\
                            .view(B,self.stage1_choose,self.lrm_pv_input_dim)
        item_input_base = self.PVInputNorm(item_input_base)

        device = item_input_base.device
        L = 6  # slate length

        index_list = [] 
        prob_list = []

        for _ in range(self.generative_list_num):
            # prefix context: aggregated projection of already selected items (B, pv_input_dim)
            prefix_context = torch.zeros(B, self.lrm_pv_input_dim, device=device)

            sampled_indices_steps = []
            sampled_logprob_steps = []

            available_mask = torch.ones(B, self.stage1_choose, dtype=torch.bool, device=device)

            for step in range(L):
                pv_ui_input = (user_input + prefix_context.unsqueeze(1)) + item_input_base  # (B, C, dim)

                pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.lrm_pv_hidden_dims)
                pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.lrm_pv_hidden_dims)
                pv_E_step = pv_ui_enc + pos_emb

                lrm_encoder_output_step = self.lrmEncoder(pv_E_step, mask = self.PV_attn_mask)

                rerank_score_step = self.lrmOutput(lrm_encoder_output_step.view(B*self.stage1_choose,self.lrm_pv_hidden_dims))\
                                            .view(B,self.stage1_choose)

                logits_masked = rerank_score_step.masked_fill(~available_mask, -1e9)

                probs_step = torch.softmax(logits_masked, dim=-1)  # (B, C)

                
                next_item = torch.multinomial(probs_step, num_samples=1)  # (B,1)

                next_item_logprob = torch.gather(torch.log(probs_step + 1e-9), 1, next_item)  # (B,1)

                sampled_indices_steps.append(next_item)          # list of (B,1)
                sampled_logprob_steps.append(next_item_logprob)  # list of (B,1)

                available_mask.scatter_(1, next_item, False)

                
                next_item_idx = next_item.squeeze(1)  # (B,)
                gathered_emb = candidate_item_emb[torch.arange(B, device=device), next_item_idx, :]  # (B, dim)
                proj_gathered = self.PVItemInputMap(gathered_emb)  # (B, dim)
                proj_gathered = self.PVInputNorm(proj_gathered)
                prefix_context = prefix_context + proj_gathered  # accumulate

            sampled_indices = torch.cat(sampled_indices_steps, dim=1)   # (B, L)
            sampled_logprobs = torch.cat(sampled_logprob_steps, dim=1)  # (B, L)

            list_logprob = torch.sum(sampled_logprobs, dim=1, keepdim=True)  # (B,1)

            
            index_list.append(sampled_indices)  # append (B,L)
            prob_list.append(list_logprob)      # append (B,1)

   
        Index_matrix = torch.stack(index_list, dim=1)
        Prob_matrix = torch.cat(prob_list, dim=1)

        
        with torch.no_grad():
            Evaluator_score = self.evaluator.forward_eval(batch,candidate_score,shuffled_candidate_indicies)
            Evaluator_score = Evaluator_score.unsqueeze(1)  # (B,1,C)

        Evaluator_score = Evaluator_score.expand(-1, self.generative_list_num, -1)  # (B,K,C)
        # gather per-list item rewards: Reward_matrix shape (B, K, L)
        Reward_matrix = torch.gather(Evaluator_score, dim=2, index=Index_matrix)
        Reward_weights = torch.tensor(
                [1.0 / math.log2(i + 1) for i in range(1, L+1)],  # i=1~L
                dtype=Prob_matrix.dtype,
                device=Reward_matrix.device  
            ) 
        Reward_matrix = Reward_matrix * Reward_weights  # broadcast (B,K,L)
        Reward_matrix = torch.sum(Reward_matrix, dim=2)  # (B,K)

        Reward_var, Reward_mean = torch.var_mean(Reward_matrix, dim=1, keepdim=True)  # (B,1),(B,1)
        Reward_epsilon = 1e-6
        Advantage_matrix = (Reward_matrix - Reward_mean) / (torch.sqrt(Reward_var) + Reward_epsilon)  # (B,K)
        Softmax_Advantage_matrix = torch.softmax(Advantage_matrix, dim=-1)  # (B,K)

       
        Prob_matrix_exp = torch.exp(Prob_matrix)  # (B,K)
        R_guide_G_loss = -torch.sum(Prob_matrix_exp * Softmax_Advantage_matrix, dim=1).mean()

        loss = self.R_guide_G_weight * R_guide_G_loss + self.G_weight * loss_G
               
        return loss
               

    def forward_eval(self, batch,candidate_score,candidate_indicies):
        candidate_indicies=candidate_indicies.long()
        candidate_item_emb= self.his_enc.iIDEmb(candidate_indicies)
        user_state=self.his_enc(batch)
        # batch size
        B = user_state.shape[0]        
        # input layer
        # (B, 1, pv_input_dim)
        user_input = self.PVUserInputMap(user_state).view(B,1,self.lrm_pv_input_dim)
        user_input = self.PVInputNorm(user_input)
        # (B, C, pv_input_dim)
        item_input = self.PVItemInputMap(candidate_item_emb.view(B*self.stage1_choose,self.lrm_pv_input_dim))\
                            .view(B,self.stage1_choose,self.lrm_pv_input_dim)
        item_input = self.PVInputNorm(item_input)
        # (B, C, pv_input_dim)
        pv_ui_input = user_input + item_input
        # (B, C, pv_enc_dim)
        pv_ui_enc = self.PVOutput(pv_ui_input).view(B,self.stage1_choose,self.lrm_pv_hidden_dims)
        # positional encoding (1, C, pv_enc_dim)
        pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(1,self.stage1_choose,self.lrm_pv_hidden_dims)
        # (B, C, pv_enc_dim)
        pv_E = pv_ui_enc + pos_emb
        
        item_input_base = self.PVItemInputMap(
            pv_E.view(B*self.stage1_choose, self.lrm_pv_input_dim)
        ).view(B, self.stage1_choose, self.lrm_pv_input_dim)
        item_input_base = self.PVInputNorm(item_input_base)

        # ====== 自回归生成 list ======
        L = 6  # list 长度
        prefix_context = torch.zeros(B, self.lrm_pv_input_dim, device=user_input.device)
        available_mask = torch.ones(B, self.stage1_choose, dtype=torch.bool, device=user_input.device)

<<<<<<< HEAD
        final_scores = torch.zeros(B, self.stage1_choose, device=user_input.device)

=======
        # 存储分数
        final_scores = torch.zeros(B, self.stage1_choose, device=user_input.device)

        # 递减分数，比如 1, 0.5, 0.33...
>>>>>>> 32d6d78 (push code)
        step_scores = torch.tensor([1.0 / (i+1) for i in range(L)], device=user_input.device)


        for step in range(L):
            # 每一步重新打分
            pv_ui_input = (user_input + prefix_context.unsqueeze(1)) + item_input_base
            pv_ui_enc = self.PVOutput(pv_ui_input).view(B, self.stage1_choose, self.lrm_pv_hidden_dims)

            pos_emb = self.PVPosEmb(self.PV_pos_emb_getter).view(
                1, self.stage1_choose, self.lrm_pv_hidden_dims
            )
            pv_E_step = pv_ui_enc + pos_emb

            lrm_encoder_output_step = self.lrmEncoder(pv_E_step, mask=self.PV_attn_mask)
            rerank_score_step = self.lrmOutput(
                lrm_encoder_output_step.view(B*self.stage1_choose, self.lrm_pv_hidden_dims)
            ).view(B, self.stage1_choose)

<<<<<<< HEAD
            logits_masked = rerank_score_step.masked_fill(~available_mask, -1e9)

            next_item = torch.argmax(logits_masked, dim=-1, keepdim=True)  # [B,1]

            final_scores.scatter_(1, next_item, step_scores[step])

            available_mask.scatter_(1, next_item, False)

=======
            # mask 已选 item
            logits_masked = rerank_score_step.masked_fill(~available_mask, -1e9)

            # 选取 top1
            next_item = torch.argmax(logits_masked, dim=-1, keepdim=True)  # [B,1]

            # 给这个位置打分 (1, 0.5, 0.33, ...)
            final_scores.scatter_(1, next_item, step_scores[step])

            # 更新 mask
            available_mask.scatter_(1, next_item, False)

            # 更新 prefix_context
>>>>>>> 32d6d78 (push code)
            gathered_emb = candidate_item_emb[torch.arange(B, device=user_input.device), next_item.squeeze(1), :]
            proj_gathered = self.PVItemInputMap(gathered_emb)
            proj_gathered = self.PVInputNorm(proj_gathered)
            prefix_context = prefix_context + proj_gathered

        return final_scores