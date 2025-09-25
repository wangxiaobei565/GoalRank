import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import pickle
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score
from evaluator import TopKMetric
# from sklearn.metrics import roc_auc_score
def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError

# import numpy as np
# from sklearn.metrics import roc_auc_score

# def dcg(relevances):
#     """计算折扣累积增益 (DCG)。"""
#     relevances = np.array(relevances)  # 确保 relevances 是一个 NumPy 数组
#     return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

# def evaluate_multi(slates, slate_labels, preds, indicies, metric_ks):
#     slates=slates.tolist()
#     slate_labels=slate_labels.tolist()
#     preds =preds.tolist() 
#     indicies=indicies.tolist()
#     ndcg, map, auc, hr, recall = [[] for _ in range(len(metric_ks))], [[] for _ in range(len(metric_ks))], [], [[] for _ in range(len(metric_ks))], [[] for _ in range(len(metric_ks))]

#     for slate, label, pred, indicie in zip(slates, slate_labels, preds, indicies):
#         # 计算 AUC
#         if len(set(label)) > 1:  # 确保有多个类别
#             auc_value = roc_auc_score(label, pred[0:6])  # 仅对 slate 的预测进行 AUC 计算
#         else:
#             auc_value = 0.5  # 或使用 None，根据需要选择默认值
#         auc.append(auc_value)
#         for i, scope in enumerate(metric_ks):
#             ideal_relevances = truncate_and_pad(sorted(label, reverse=True), scope)
#             # 获取 top_k 的预测

#             top_k_indices = [index for _, index in sorted(zip(pred, indicie), key=lambda x: x[0], reverse=True)][0:scope]
#             actual_relevances = [0 for j in top_k_indices]
#             selected_ids = [slate[i] for i in range(len(label)) if label[i] == 1]
#             for j in top_k_indices:
#                 if j in selected_ids:  # 如果在 binary_label 中的值为 1
#                     actual_relevances[j] = 1  # 对应位置置为 1


#             # 计算 DCG
#             actual_dcg = dcg(actual_relevances)
#             ideal_dcg = dcg(ideal_relevances)

#             # 计算 NDCG
#             _ndcg = float(actual_dcg) / ideal_dcg if ideal_dcg != 0 else 0.
#             ndcg[i].append(_ndcg)

#             # 计算 MAP
#             precision_at_k = []
#             for rank, index in enumerate(top_k_indices):
#                 if index in selected_ids:
#                     precision_at_k.append(sum(actual_relevances[:rank + 1]) / (rank + 1))  # 当前精确度
#             map_value = np.mean(precision_at_k) if precision_at_k else 0.0
#             map[i].append(map_value)

#             relevant_retrieved = sum(1 for j in top_k_indices if j in selected_ids)  # 被检索到的相关项数量
#             total_relevant = len(selected_ids)  # 总的相关项数量
#             recall_value = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
#             recall[i].append(recall_value)

#             # 计算 Hit Ratio
#             hit_ratio_value = 1.0 if relevant_retrieved > 0 else 0.0  # 如果有相关项被检索到，记为 1
#             hr[i].append(hit_ratio_value)

#     return {
#         "map": np.mean(np.array(map), axis=-1),
#         "ndcg": np.mean(np.array(ndcg), axis=-1),
#         "auc": np.mean(auc),
#         "hit_ratio": np.mean(np.array(hr), axis=-1),
#         "recall": np.mean(np.array(recall), axis=-1),
#     }

def truncate_and_pad(seq, length):
    """截断并填充序列到给定长度。"""
    if len(seq) > length:
        return seq[:length]
    else:
        return seq + [0] * (length - len(seq))




def model_train(tra_data_loader,  test_data_loader, model, args, logger,item_score,device):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)
    best_auc=0
  
   
    # flag=0
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()
        total_loss = 0
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            loss= model.forward(train_batch)  
            loss.backward()
            total_loss = total_loss + loss.item()
            optimizer.step()
            # if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
        print('Loss in epoch: %.4f' % ( total_loss/len(tra_data_loader)))
        logger.info('Loss in epoch: %.4f' % ( total_loss/len(tra_data_loader)))

        
        
        lr_scheduler.step()



        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model.eval()
            with torch.no_grad():
                aucs=[]
                auc=0
           

                # metrics_dict_mean = {}
                for i,test_batch in enumerate(test_data_loader):
                    test_batch = [x.to(device) for x in test_batch]
                    test_pred= model.forward_eval(test_batch) 
                    # label=torch.cat((test_batch[4],torch.zeros(test_batch[4].size(0),args.stage1_choose-test_batch[4].size(1)).cuda()),dim=1)

                    # MT=TopKMetric(args.metric_ks,args.stage1_choose,test_batch[4],test_pred)
                    # test_metrics=MT.get_metrics()
                    aucs.append(roc_auc_score(test_batch[2].cpu().numpy().flatten(), test_pred.cpu().numpy().flatten()))

                    # for k, v in test_metrics.items():
                    #     if k not in test_metrics_dict:
                    #         test_metrics_dict[k] = []
                    #     else:
                    #         test_metrics_dict[k].append(v)
                        
            # test_metrics={key: total / len(test_data_loader) for key, total in test_metrics_dict.items()}
            # flag+=1
            auc = round(np.mean(aucs), 4)
            if auc> best_auc:
                best_auc=auc
                torch.save(model, args.dataset+'_stage1.pth')

                
                
        
                
            print('Test------------------------------------------------------')
            logger.info('Test------------------------------------------------------')
            print(auc)
            logger.info(auc)
            # if flag==0:
            print('Best Test---------------------------------------------------------')
            logger.info('Best Test---------------------------------------------------------')
            print(best_auc)
            logger.info(best_auc)
        
            # if flag>args.patience:
            #         break                  
            
    
        
    # if args.eval_interval > epochs:
    #     best_model = copy.deepcopy(model)
    
    
    # top_100_item = []
    # with torch.no_grad():
    #     test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
    #     test_metrics_dict_mean = {}
    #     for test_batch in test_data_loader:
    #         test_batch = [x.to(device) for x in test_batch]
    #         rep_diffu= best_model(test_batch[0], test_batch[1], train_flag=False)
    #         scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)   ### Inner Production
    #         # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing
            
    #         _, indices = torch.topk(scores_rec_diffu, k=100)
    #         top_100_item.append(indices)

    #         metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
    #         for k, v in metrics.items():
    #             test_metrics_dict[k].append(v)
    
    # for key_temp, values_temp in test_metrics_dict.items():
    #     values_mean = round(np.mean(values_temp) * 100, 4)
    #     test_metrics_dict_mean[key_temp] = values_mean
    # print('Test------------------------------------------------------')
    # logger.info('Test------------------------------------------------------')
    # print(test_metrics_dict_mean)
    # logger.info(test_metrics_dict_mean)
    # print('Best Eval---------------------------------------------------------')
    # logger.info('Best Eval---------------------------------------------------------')
    # print(best_metrics_dict)
    # print(best_epoch)
    # logger.info(best_metrics_dict)
    # logger.info(best_epoch)

    print(args)

            

    return loss
    
