from utils import *
import pickle
import numpy as np
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import KFold
import os
from tqdm import tqdm
import scipy
from scipy.sparse import csr_matrix, lil_matrix
import heapq
from config import *

with open(groups_txt, 'rb') as text:
    Groups = pickle.load(text)
with open(members_txt, 'rb') as text:
    members = pickle.load(text)
with open(events_txt, 'rb') as text:
    events = pickle.load(text)


def JudgeModel(realScore, estScore, threshold):
    if realScore <= no_threshold and estScore <= threshold:
        return True
    elif realScore >= yes_threshold and estScore >= threshold:
        return True
    return False

#     return True
# if realScore <= maybe_threshold and estScore <= maybe_threshold:
    # elif realScore >= yes_threshold and estScore >= yes_threshold:
    #     return True
    # elif maybe_threshold < realScore < yes_threshold and maybe_threshold < estScore < yes_threshold:
    #     return True
    # return False


# 构建member_2_id字典
if os.path.exists('member_2_id.npy'):
    member_2_id = np.load('member_2_id.npy', allow_pickle=True).item()
else:
    member_2_id = {}
    for i in range(len(members)):
        member_2_id[members[i].id] = i
    np.save('member_2_id.npy', member_2_id)


# 构建event_2_id字典
if os.path.exists('event_2_id.npy'):
    event_2_id = np.load('event_2_id.npy', allow_pickle=True).item()
else:
    event_2_id = {}
    for i in range(len(events)):
        event_2_id[events[i].id] = i
    np.save('event_2_id.npy', event_2_id)

# 构建event_2_group字典
if os.path.exists('event_2_group.npy'):
    event_2_group = np.load('event_2_group.npy', allow_pickle=True).item()
else:
    event_2_group = {}
    for i in range(len(events)):
        event_2_group[events[i].id] = events[i].group
    np.save('event_2_group.npy', event_2_group)

# 导入用户-活动评价矩阵
if os.path.exists('members_2_events.npz'):
    mem_events = scipy.sparse.load_npz('members_2_events.npz').tolil()
else:
    mem_events = lil_matrix((len(members), len(events)))
    for j in range(len(events)):
        for i in range(len(members)):
            score = 0
            if members[i].id in events[j].yes_members:
                score = 3
            if members[i].id in events[j].maybe_members:
                score = 2
            if members[i].id in events[j].no_members:
                score = 1
            mem_events[i, j] = score
        if j % 100 == 0:
            print('正在处理第{}/{}个成员'.format(j, len(members)))
    scipy.sparse.save_npz('members_2_events', mem_events.tocsr())

# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
# 定义数据划分方式
kf = KFold(n_splits=5, shuffle=shuffle, random_state=1)
# 定义模型
algo = SVD()

for trainset, testset in kf.split(data):

    # 生成训练集上的用户-评价矩阵
    train_mem_events = mem_events.copy()
    for obj in testset:
        mem_site, event_site, score = member_2_id[obj[0]], event_2_id[obj[1]], obj[2]
        # 将测试集上的位置置0
        train_mem_events[mem_site, event_site] = 0

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # 定义误差
    right = 0
    total = 0

    # 基于成员主题
    mem_2_topics = scipy.sparse.load_npz('topics\\members_2_topics.npz').tolil()

    # 将成员的计算的阈值导出
    result = []

    for obj in tqdm(predictions):

        mem_site, event_site = member_2_id[obj.uid], event_2_id[obj.iid]
        # 计算参加了这个活动的成员.由于之后是在训练集上计算，所以添加了某些测试集上的成员也没问题
        members_site = []
        for i in events[event_site].yes_members + events[event_site].no_members + events[event_site].maybe_members:
            if i != 'null':
                members_site.append(member_2_id[i])

        if net_way == 'topics':
            # 计算与该成员主题相似的
            k_nearest = []
            for i in members_site:
                same_topics = len(set(members[mem_site].topics) & set(members[i].topics))
                k_nearest.append(same_topics)
            # 计算最相似前k个成员位置
            max_num_index = list(map(k_nearest.index, heapq.nlargest(top_k, k_nearest)))
            # 计算前k个成员对该成员参加活动的影响
            # 如果去，阈值降低；不去，阈值不变(这个有问题，后续可以改进)
            effect_threshold = 0
            for i in max_num_index:
                if need_weight:
                    if train_mem_events[members_site[i], event_site] == 3:
                        effect_threshold += k_nearest[i]
                    elif train_mem_events[members_site[i], event_site] == 2:
                        effect_threshold += 0
                    else:
                        effect_threshold -= k_nearest[i]
                else:
                    effect_threshold += train_mem_events[members_site[i], event_site] - 2
            # 计算受影响后的阈值
            threshold = basic_threshold - alpha_topics * effect_threshold   # 除上该成员的主题(可选)

        elif net_way == 'events':
            # 计算与该成员活动相似的
            k_nearest = []
            # 计算该成员参加的活动
            events_site = train_mem_events.rows[mem_site]
            for i in members_site:
                i_events_site = train_mem_events.rows[i]
                same_events = len(set(events_site) & set(i_events_site))
                k_nearest.append(same_events)
            # 计算最相似前k个成员位置
            max_num_index = list(map(k_nearest.index, heapq.nlargest(top_k, k_nearest)))
            # 计算前k个成员对该成员参加活动的影响
            effect_threshold = 0
            for i in max_num_index:
                if train_mem_events[members_site[i], event_site] == 3:
                    effect_threshold += k_nearest[i]
                elif train_mem_events[members_site[i], event_site] == 2:
                    effect_threshold += 0
                else:
                    effect_threshold -= k_nearest[i]
            # 计算受影响后的阈值
            threshold = basic_threshold - alpha_events * effect_threshold

        elif net_way == 'groups':
            # 计算与该成员社团相似的
            k_nearest = []
            # 计算该成员参加的社团
            events_site = train_mem_events.rows[mem_site]
            groups = []
            for t in events_site:
                groups.append(event_2_group[events[t].id])
            for i in members_site:
                i_events_site = train_mem_events.rows[i]
                i_groups = []
                for t in i_events_site:
                    i_groups.append(event_2_group[events[t].id])
                same_groups = len(set(groups) & set(i_groups))
                k_nearest.append(same_groups)
            # 计算最相似前k个成员位置
            max_num_index = list(map(k_nearest.index, heapq.nlargest(top_k, k_nearest)))
            # 计算前k个成员对该成员参加活动的影响
            effect_threshold = 0
            for i in max_num_index:
                # effect_threshold += train_mem_events[members_site[i], event_site] * k_nearest[i]
                if train_mem_events[members_site[i], event_site] == 3:
                    effect_threshold += k_nearest[i]
                elif train_mem_events[members_site[i], event_site] == 2:
                    effect_threshold += 0
                else:
                    effect_threshold -= k_nearest[i]
            # 计算受影响后的阈值
            threshold = basic_threshold - alpha_groups * effect_threshold

        else:  # 不使用网络的结果
            threshold = basic_threshold

        # 网络演化
        if net_growth:
            train_mem_events[mem_site, event_site] = obj.r_ui

        # 统计结果
        result.append([obj.uid, obj.iid, obj.r_ui, obj.est, threshold])
        right += JudgeModel(obj.r_ui, obj.est, threshold)
        total += 1

    # 将结果写入txt
    with open('result_' + net_way + '.txt', 'a+', encoding='utf-8') as f:
        for data in result:
            for ele in data:
                f.write(str(ele))
                f.write('\t')
            f.write('\n')
        f.close()
    print(right / total)
    break


