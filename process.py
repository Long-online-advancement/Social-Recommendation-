from utils import *
import pickle
import numpy as np
import scipy
from scipy.sparse import csr_matrix, lil_matrix
import os
from tqdm import tqdm

with open(groups_txt, 'rb') as text:
    groups = pickle.load(text)
with open(members_txt, 'rb') as text:
    members = pickle.load(text)
with open(events_txt, 'rb') as text:
    events = pickle.load(text)

topics_num = 17128


# 生成主题的txt
if not os.path.exists('topics\\topics.txt'):
    # 开始计算主题总数
    topics_groups = set()
    topics_members = set()
    for i in tqdm(range(len(groups))):
        topics_groups = topics_groups | set(groups[i].topics)
    for i in tqdm(range(len(members))):
        topics_members = topics_members | set(members[i].topics)
    topics = topics_groups | topics_members
    print('社团共有的主题活动为:{}'.format(len(topics_groups)))
    print('成员共有的主题活动为:{}'.format(len(topics_members)))
    print('总共出现的主题为:{}'.format(len(topics)))
    with open('topics\\topics_groups.txt', 'a+', encoding='utf-8') as f:
        for data in topics_groups:
            f.write(data + '\n')
        f.close()

    with open('topics\\topics_members.txt', 'a+', encoding='utf-8') as f:
        for data in topics_members:
            f.write(data + '\n')
        f.close()

    with open('topics\\topics.txt', 'a+', encoding='utf-8') as f:
        for data in topics:
            f.write(data + '\n')
        f.close()

else:
    f = open("topics\\topics.txt", "r")
    topics = f.readlines()
    f.close()

# 生成矩阵 成员-主题
if os.path.exists('topics\\members_2_topics.npz'):
    mem_2_topics = scipy.sparse.load_npz('topics\\members_2_topics.npz').tolil()
else:
    mem_2_topics = lil_matrix((len(members), len(topics)))
    for i in tqdm(range(len(members))):
        for j in range(len(topics)):
            mem_2_topics[i, j] = topics[j] in members[i].topics
    scipy.sparse.save_npz('topics\\members_2_topics.npz', mem_2_topics.tocsr())

# 生成矩阵 社团-主题
if os.path.exists('topics\\groups_2_topics.npz'):
    groups_2_topics = scipy.sparse.load_npz('topics\\groups_2_topics.npz').tolil()
else:
    groups_2_topics = lil_matrix((len(groups), len(topics)))
    for i in tqdm(range(len(groups))):
        for j in range(len(topics)):
            groups_2_topics[i, j] = topics[j] in groups[i].topics
    scipy.sparse.save_npz('topics\\groups_2_topics.npz', groups_2_topics.tocsr())


# 观察数据分布情况，是否需要三分类
def numbers_decision():
    yes_member = 0
    no_member = 0
    maybe_member = 0

    for i in tqdm(range(len(events))):
        yes_member = yes_member + len(events[i].yes_members)
        no_member = no_member + len(events[i].no_members)
        maybe_member = maybe_member + len(events[i].maybe_members)

    print('回答YES的总次数为{}'.format(yes_member))
    print('回答NO的总次数为{}'.format(no_member))
    print('回答MAYBE的总次数为{}'.format(maybe_member))


numbers_decision()

# # 构建成员主题相同数目的完整矩阵 TODO：太麻烦了，时间太长,不合适
# if os.path.exists('mem_2_mem_topics.npz'):
#     groups_2_topics = scipy.sparse.load_npz('mem_2_mem_topics.npz')
# else:
#     mem_mem_topics = lil_matrix((len(members), len(members)))
#     for i in range(len(members)):
#         for j in tqdm(range(i + 1, len(members))):
#             mem_mem_topics[i, j] = np.sum(mem_2_topics[i, :] * (mem_2_topics[j, :].transpose()))
#     scipy.sparse.save_npz('mem_2_mem_topics.npz', mem_mem_topics.tocsr())


