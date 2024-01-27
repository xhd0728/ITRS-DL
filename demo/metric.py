import numpy as np


# 指标计算
def itm_eval(scores_t2i, txt2img):

    ranks = np.zeros(scores_t2i.shape[0])
    MRRs = np.zeros(scores_t2i.shape[0])
    MAPs = np.zeros(scores_t2i.shape[0])

    Top_K_NDCG = [1, 3, 5, 10]
    Top_K_Precision = [1, 3, 5, 10]
    NDCGs = np.zeros((len(Top_K_NDCG), scores_t2i.shape[0]))
    Precisons = np.zeros((len(Top_K_Precision), scores_t2i.shape[0]))

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        # 计算R@k和MRRs
        rank = 1e20
        sort_rank = []
        ap = 0
        for i, r in enumerate(txt2img[index]):
            tmp = np.where(inds == r)[0][0]
            sort_rank.append(tmp)
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        MRRs[index] = 1.0 / (rank + 1)

        # 计算MAP
        sort_rank = sorted(sort_rank)
        ap = [1.0 * (i + 1) / (x + 1) for i, x in enumerate(sort_rank)]
        ap = sum(ap) / len(sort_rank)
        MAPs[index] = ap

        # 计算NDCG
        for j, k in enumerate(Top_K_NDCG):
            dcg = 0.0
            idcg = np.sum(1.0 / np.log2(np.arange(2, k + 2)))
            for i, r in enumerate(inds[:k]):
                if r in txt2img[index]:
                    dcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg
            NDCGs[j][index] = ndcg

        # 计算精确率
        for j, k in enumerate(Top_K_Precision):
            find_list = np.intersect1d(txt2img[index], inds[:k])
            precision = len(find_list) / k

            Precisons[j][index] = precision

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ir_mean = (ir1 + ir3 + ir5 + ir10) / 4

    MRR = np.mean(MRRs)
    MAP = np.mean(MAPs)
    NDCG_mean = np.mean(NDCGs, axis=1)
    Precision_mean = np.mean(Precisons, axis=1)

    result = {
        'R@1': ir1, 'R@3': ir3, 'R@5': ir5, 'R@10': ir10, 'Avg_R': ir_mean,
        'MRR': MRR,
        'MAP': MAP
    }
    for i, ndcg in enumerate(NDCG_mean):
        result.setdefault(f"NDCG@{Top_K_NDCG[i]}", ndcg)
    for i, p in enumerate(Precision_mean):
        result.setdefault(f"Precision@{Top_K_Precision[i]}", p)

    return result


def compute_rr(true_label, predicted_labels):
    """
    计算查询单个标签对应的平均倒数排名

    :param true_label: 单个真实标签, (1, )
    :param predicted_labels: 单真实标签对应的topk个预测标签列表, (1, topk)
    :return: 单标签对应的平均倒数排名
    """
    k = len(predicted_labels)
    rr = []
    count = 1
    for i in range(k):
        if predicted_labels[i] == true_label:
            rr.append((1.0 / (i + 1)) * count)
            count += 1
    return np.mean(rr) if len(rr) > 0 else 0


def compute_mrr(true_labels_list, predicted_labels_list):
    """
    计算一批标签对应的平均倒数排名

    :param true_labels_list: 真实标签列表 (n_queries, )
    :param predicted_labels_list: 真实标签列表分别对应的topk个预测标签, (n_queries, topk)
    :return: 一批标签对应的平均倒数排名
    """
    mrr = []
    for true_label, predicted_labels in zip(true_labels_list, predicted_labels_list):
        rr = compute_rr(true_label, predicted_labels)
        mrr.append(rr)
    return np.mean(mrr) if len(mrr) > 0 else 0


def MRR(scores_t2i, txt2img):
    """
    :param scores_t2i:
    :param txt2img:
    :return:
    """
    # 计算每个query的MRR
    query_nums = scores_t2i.shape[0]
    mrr = np.zeros(query_nums)
    for i in range(query_nums):
        # 计算每个query的MRR
        rank = 0
        for j in range(scores_t2i.shape[1]):
            if txt2img[i] == scores_t2i[i][j]:
                rank = j + 1
                break
        mrr[i] = 1.0 / rank if rank != 0 else 0
    # 计算平均MRR
    mrr_mean = np.mean(mrr)
    return mrr_mean


def NDCG(scores_t2i, txt2img):
    """
    :param scores_t2i:
    :param txt2img:
    :return:
    """
    # 计算每个query的NDCG
    query_nums = scores_t2i.shape[0]
    ndcg = np.zeros(query_nums)
    for i in range(query_nums):
        # 计算每个query的NDCG
        rank = 0
        dcg = 0
        for j in range(scores_t2i.shape[1]):
            if txt2img[i] == scores_t2i[i][j]:
                rank = j + 1
                break
        for j in range(rank):
            dcg += 1.0 / np.log2(j + 2)
        # 计算每个query的NDCG
        idcg = 0
        for j in range(rank):
            idcg += 1.0 / np.log2(j + 2)
        ndcg[i] = dcg / idcg if idcg != 0 else 0
    # 计算平均NDCG
    ndcg_mean = np.mean(ndcg)
    return ndcg_mean


def mAP(scores_t2i, txt2img):
    """
    :param scores_t2i:
    :param txt2img:
    :return:
    """
    # 计算每个query的mAP
    query_nums = scores_t2i.shape[0]
    mAP = np.zeros(query_nums)
    for i in range(query_nums):
        # 计算每个query的mAP
        rank = 0
        for j in range(scores_t2i.shape[1]):
            if scores_t2i[i][j] == txt2img[i]:
                rank += 1
                mAP[i] = rank / (j + 1)
                break
    # 计算平均mAP
    mAP_mean = np.mean(mAP)
    return mAP_mean
