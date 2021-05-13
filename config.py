# 不同阈值
no_threshold = 1.8
yes_threshold = 2.3
basic_threshold = 2.0

# 指定文件所在路径
file = ['score.txt', 'score_sorted.txt']
file_path = file[0]

# 是否打乱数据集
shuffle = True

# 选择网络中影响最大的前k个
top_k = 2

# 超参数
alpha_topics = 0.001
alpha_events = 0.001
alpha_groups = 0.005

# 网络构建方法
way = ['no_net', 'topics', 'events', 'groups']
net_way = way[1]

# 网络是否需要加权
need_weight = True

# 网络是否演化
net_growth = False
