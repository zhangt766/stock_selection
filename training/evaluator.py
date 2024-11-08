import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(prediction, ground_truth, mask, tickers=None, report=False, top_k=[1]):
    """
    Args:
        prediction: 预测值
        ground_truth: 真实值
        mask: 数据掩码
        tickers: 股票代码列表
        report: 是否打印信息
        top_k: 要查看的top-k列表，如[1], [5], [10], [1,5], [1,5,10]
    """
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    
    # 动态初始化bt_long字典
    bt_dict = {k: 1.0 for k in top_k}
    selected_stocks = []

    for i in range(prediction.shape[1]):
        # 获取预测排名
        rank_pre = np.argsort(prediction[:, i])
        pre_tops = {k: set() for k in top_k}
        
        # 选择top-k的股票
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            for k in top_k:
                if len(pre_tops[k]) < k:
                    pre_tops[k].add(cur_rank)

        # 记录每天的选股信息
        day_info = {'day': i}
        for k in top_k:
            day_info[f'top{k}'] = {
                'stocks': [tickers[idx] if tickers is not None else f"Stock_{idx}" for idx in pre_tops[k]],
                'pred_returns': [prediction[idx][i] for idx in pre_tops[k]],
                'actual_returns': [ground_truth[idx][i] for idx in pre_tops[k]]
            }
        selected_stocks.append(day_info)

        # 计算每个top-k的回测收益
        for k in top_k:
            real_ret_rat = 0
            for pre in pre_tops[k]:
                real_ret_rat += ground_truth[pre][i]
            real_ret_rat /= k  # 平均收益率
            bt_dict[k] += real_ret_rat

        # 计算MRR (只针对top1)
        if 1 in top_k:
            rank_gt = np.argsort(ground_truth[:, i])
            top1_pos_in_gt = 0
            for j in range(1, prediction.shape[0] + 1):
                cur_rank = rank_gt[-1 * j]
                if mask[cur_rank][i] < 0.5:
                    continue
                else:
                    top1_pos_in_gt += 1
                    if cur_rank in pre_tops[1]:
                        break
            if top1_pos_in_gt == 0:
                all_miss_days_top += 1
            else:
                mrr_top += 1.0 / top1_pos_in_gt

    # 计算性能指标
    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top) if 1 in top_k else 0
    for k in top_k:
        performance[f'btl{k}'] = bt_dict[k]

    # 打印选股结果
    if report:
        print("\n================== Stock Selection Results ==================")
        for k in top_k:
            print(f"\nTop {k} Selection:")
            print(f"Day | {'Stocks':30s} | {'Predicted Returns':30s} | {'Actual Returns':30s}")
            print("-" * 100)
            
            for day_info in selected_stocks:
                stocks = day_info[f'top{k}']['stocks']
                pred_rets = day_info[f'top{k}']['pred_returns']
                act_rets = day_info[f'top{k}']['actual_returns']
                
                stocks_str = ', '.join(stocks)
                pred_rets_str = ', '.join([f"{x:.4f}" for x in pred_rets])
                act_rets_str = ', '.join([f"{x:.4f}" for x in act_rets])
                
                print(f"Day {day_info['day']:3d} | {stocks_str:30s} | {pred_rets_str:30s} | {act_rets_str:30s}")

        print("\n================== Evaluation Results ==================")
        print(f"MSE: {performance['mse']:.6f}")
        if 1 in top_k:
            print(f"MRRT: {performance['mrrt']:.6f}")
        for k in top_k:
            print(f"Total Return (Top{k}): {bt_dict[k] - 1.0:.4f}")

    return performance