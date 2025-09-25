# GoalRank



# 0.Setup

```
conda create -n GoalRank python=3.8
conda activate GoalRank
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
python -m ipykernel install --user --name GoalRank --display-name "GoalRank"
```

# 1. Data Setup

### Data Processing

- filter data by 20-core for cleaned data
- parpare item-wise stage1 data for retriever
- parpare list-structure stage2 data for Ranker




# 2. Run code

## 2.1 Pretraining

```
bash train_{data_set}_stage1.sh
```

## 2.2 Training

- Train Normal Method

```
bash train_{model name}_{data_set}_stage2.sh
```
- Train GoalRank
 1. Train Evaluator 
```
bash train_Evaluator_{data_set}_stage2.sh
```
 2. Train GoalRank
```
# add parser in GorlRank training scripts
# parser.add_argument('--generative_list_num', default=100, help='generative_list_num_weight')
# parser.add_argument('--path_evaluator', default='Evaluator/evaluator_kuai.pth', help='path_evaluator')
bash train_GoalRank_{data_set}_stage2.sh
```

## 2.3 Log

We report Ratio@L (H@L), NDCG@L (N@L), MAP@L (M@L), F1@L, and AUC 





# 3. Our code

We have released our GoalRank training approach : 


1. Train a Evalautor, Supposed we have a well-trained Evaluator(Reward Model).
```
self.evaluator = Evaluator(args)
evaluator_state_dict = torch.load(args.path_evaluator)
self.evaluator.load_state_dict(evaluator_state_dict)
self.evaluator.eval()
self.evaluator.to(args.device)
```
2. Group Construction by several ways, For example, we use  Softmax-based stochastic sampling
```
## Softmax-based stochastic sampling
def prob_transform_matrix(matrix, bs):
    ones = torch.ones([bs, 1], device=matrix.device)  
    cumsum = torch.cumsum(matrix[:, :5], dim=1)  
    result = ones - cumsum
    final_result = torch.cat([ones, result], dim=1) 
    return final_result

rank_socre = self.Generator(inputs)
score_softmax = torch.softmax(rank_socre, dim=-1)
        
index_list = []
prob_list = []
for index in range(self.generative_list_num):

    uni = torch.rand_like(score_softmax)
    exponent = 1.0 / (score_softmax + 1e-9)
    exponent = torch.clamp(exponent, -1e9, 1e9)
    tem_output = torch.pow(uni, exponent)
    _, sampled_indices = torch.topk(tem_output, k=6)   
    sampled_prob = torch.gather(score_softmax, dim=-1, index=sampled_indices)
    sampled_prob = torch.maximum(sampled_prob, torch.tensor(1e-9, device=sampled_prob.device))
    adjust_sampled_prob = prob_transform_matrix(sampled_prob, B)
    adjust_sampled_prob = torch.maximum(adjust_sampled_prob, torch.tensor(1e-9, device=adjust_sampled_prob.device))
    sampled_prob_log = torch.log(sampled_prob)
    adjust_sampled_prob_log = torch.log(adjust_sampled_prob)
    list_prob = torch.sum(sampled_prob_log, dim=1, keepdim=True) - torch.sum(adjust_sampled_prob_log, dim=1, keepdim=True)  # [bs, 1]
    index_list.append(sampled_indices)
    prob_list.append(list_prob)
```  
2. Bulid Reference Model by Reward and Lists
```
with torch.no_grad():
            
    Evaluator_score = self.evaluator.forward_eval(batch,candidate_score,shuffled_candidate_indicies)
    Evaluator_score = Evaluator_score.unsqueeze(1)
   
Prob_matrix = torch.cat(prob_list, dim=1) #[bs, k]
Index_matrix = torch.stack(index_list, dim=1) #[bs, k, len_list]


Evaluator_score = Evaluator_score.expand(-1, self.generative_list_num, -1)

Reward_matrix = torch.gather(Evaluator_score, dim=2, index=Index_matrix)
Reward_weights = torch.tensor(
        [1.0 / math.log2(i + 1) for i in range(1, 7)],  # i=1~6
        dtype=Prob_matrix.dtype,
        device=Reward_matrix.device  
    ) 
Reward_matrix = Reward_matrix * Reward_weights
Reward_matrix = torch.sum(Reward_matrix, dim=2)

Reward_var, Reward_mean = torch.var_mean(Reward_matrix, dim=1, keepdim=True)

Reward_epsilon = 1e-6
Reference_matrix = torch.softmax((Reward_matrix - Reward_mean) / (torch.sqrt(Reward_var) + Reward_epsilon), dim=-1)
```
3. Caculate Loss
```
loss_G = XXX
Reference_loss = -torch.sum(Prob_matrix * Reference_matrix)
loss = self.Reference_weight * Reference_loss + self.G_weight * loss_G
```
