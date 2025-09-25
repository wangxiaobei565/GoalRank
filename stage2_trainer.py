import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import os
from evaluator import TopKMetric
from torch.cuda.amp import autocast

# ================= 数值稳定性设置（A100） =================
# 做一次对照更稳：如需性能可改 True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-6,           # ★ BF16/Adam 数值稳定关键
            betas=(0.9, 0.98)
        )
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=getattr(args, "momentum", 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _finite_grads(model, is_parallel: bool) -> bool:
    mdl = model.module if is_parallel else model
    for p in mdl.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def model_train(tra_data_loader, test_data_loader, stage1_model, model, args, logger, item_score, device):
    """
    期望 args:
      - epochs, lr, weight_decay, optimizer
      - decay_step, eval_interval, metric_ks, stage1_choose
      - clip_grad_para
      - if_use_bf16 (bool)
      - num_gpu, device, save_evaluator, evaluator_path
    """
    epochs = args.epochs
    device = args.device

    model = model.to(device)
    stage1_model = stage1_model.to(device).eval()

    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)

    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=0.1
    )

    best_metrics_dict = {}
    best_epoch = {}

    for epoch_temp in range(epochs):
        print(f'Epoch: {epoch_temp}')
        logger.info(f'Epoch: {epoch_temp}')
        model.train()
        total_loss = 0.0

        for index_temp, train_batch in enumerate(tra_data_loader):
            # 移动到设备（浮点可直接保持 fp32，autocast 会负责 downcast）
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad(set_to_none=True)

            if getattr(args, "if_use_bf16", True):
                # ---- BF16 计算图（无需 GradScaler）----
                with autocast(dtype=torch.bfloat16):
                    # 一阶段分数通常不需要梯度
                    with torch.no_grad():
                        candidate_score, candidate_indicies = stage1_model.predict_score(train_batch)
                    loss = model.forward(train_batch, candidate_score, candidate_indicies)
            else:
                # 纯 FP32 对照
                with torch.no_grad():
                    candidate_score, candidate_indicies = stage1_model.predict_score(train_batch)
                loss = model.forward(train_batch, candidate_score, candidate_indicies)

            loss.backward()

            # 裁剪 + 有限性检查
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.clip_grad_para
            )

            if _finite_grads(model, is_parallel):
                optimizer.step()
            else:
                # 梯度不有限，跳过本次 step，避免把 NaN/Inf 写入参数
                optimizer.zero_grad(set_to_none=True)
                logger.warning(f"[Epoch {epoch_temp} | Batch {index_temp}] skip step due to non-finite grads. total_norm={float(total_norm):.6f}")

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(tra_data_loader))
        print('Loss in epoch: %.4f' % avg_loss)
        logger.info('Loss in epoch: %.4f' % avg_loss)

        lr_scheduler.step()

        # ================= 验证评估 =================
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model.eval()
            with torch.no_grad():
                test_metrics_dict = {}

                for i, test_batch in enumerate(test_data_loader):
                    processed_test_batch = []
                    for x in test_batch:
                        x = x.to(device)
                        # 在推理里也用 BF16 计算可进一步稳/快（对浮点张量）
                        if getattr(args, "if_use_bf16", True) and x.is_floating_point():
                            processed_test_batch.append(x.to(torch.bfloat16))
                        else:
                            processed_test_batch.append(x)
                    test_batch = processed_test_batch

                    if getattr(args, "if_use_bf16", True):
                        with autocast(dtype=torch.bfloat16):
                            candidate_score, candidate_indicies = stage1_model.predict_score(test_batch)
                            test_pred = model.forward_eval(test_batch, candidate_score, candidate_indicies)
                    else:
                        candidate_score, candidate_indicies = stage1_model.predict_score(test_batch)
                        test_pred = model.forward_eval(test_batch, candidate_score, candidate_indicies)

                    MT = TopKMetric(args.metric_ks, args.stage1_choose, test_batch[4], test_pred)
                    test_metrics = MT.get_metrics()
                    for k, v in test_metrics.items():
                        test_metrics_dict.setdefault(k, []).append(v)

            final_metrics = {}
            for key_temp, vals in test_metrics_dict.items():
                test_mean = round(float(np.mean(vals)), 4)
                final_metrics[key_temp] = test_mean

                # 记录 best
                if key_temp not in best_metrics_dict or test_mean > best_metrics_dict[key_temp]:
                    best_metrics_dict[key_temp] = test_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    if getattr(args, "save_evaluator", False):
                        torch.save(model.state_dict(), args.evaluator_path)

            print('Test------------------------------------------------------')
            logger.info('Test------------------------------------------------------')
            print(final_metrics)
            logger.info(final_metrics)
            print('Best Test---------------------------------------------------------')
            logger.info('Best Test---------------------------------------------------------')
            print(best_metrics_dict)
            print(best_epoch)
            logger.info(best_metrics_dict)
            logger.info(best_epoch)

    print(args)
    return avg_loss
