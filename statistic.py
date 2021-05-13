from config import *

def JudgeModel(realScore, estScore, threshold):
    if realScore <= no_threshold and estScore <= threshold:
        return True
    elif realScore >= yes_threshold and estScore >= threshold:
        return True
    return False

# 导入关于主题网络的结果文件
f = open("result_no_net.txt", "r")
tmp = f.readlines()
f.close()
result = []
for line in tmp:
    str_result = line.split('\t')
    str_result[2], str_result[3], str_result[4] = float(str_result[2]), float(str_result[3]), float(str_result[4])
    result.append(str_result)

# 导入关于主题网络的结果文件
f = open("result_topics.txt", "r")
tmp = f.readlines()
f.close()
result_topics = []
for line in tmp:
    str_result = line.split('\t')
    str_result[2], str_result[3], str_result[4] = float(str_result[2]), float(str_result[3]), float(str_result[4])
    result_topics.append(str_result)

# 比较基于网络的模型提高预测正确的数量及位置
# 1.计算不同网络正确的位置
true_site = []
true_site_topic = []
for i in range(len(result_topics)):
    if JudgeModel(result[i][2], result[i][3], result[i][4]):
        true_site.append(i)
    if JudgeModel(result_topics[i][2], result_topics[i][3], result_topics[i][4]):
        true_site_topic.append(i)
# 2.计算不同的位置
# 计算网络多算对的位置
common = set(true_site) & set(true_site_topic)
better_topics_site = list(set(true_site_topic) - common)
better_site = list(set(true_site) - common)

print(len(better_topics_site), len(result_topics))
print(len(better_site), len(result_topics))
print(len(common))
