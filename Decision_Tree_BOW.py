import re
import time
import joblib
import pandas as pd
from sklearn import tree
import jieba.posseg as psg
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import export_graphviz

# 在程序开始时加载停用词列表
with open('./stoplist.txt', 'r', encoding='UTF-8') as stop_path:
    stop_words = set(stop_path.read().split())

def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            dataset = pd.read_csv(file_path, encoding='gb18030')
        except UnicodeDecodeError:
            return None, "无法读取数据集，请检查文件编码。"
    return dataset, None

def preprocess_and_clean_data(data_frame):
    # 初始化一个空的DataFrame来存储处理后的数据
    processed_data = pd.DataFrame()
    for index, row in data_frame.iterrows():
        # 清洗文本，只保留中文字符
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', row['review'])
        # 使用jieba进行分词，保留词语和词性
        words_pos = [(x.word, x.flag) for x in psg.cut(cleaned_text) if x.word not in stop_words and x.word.strip()]
        # 创建一个临时列表来存储当前review的处理结果
        temp_data_list = []
        for word, flag in words_pos:
            temp_data_list.append({
                '分词索引': index + 1,
                '词语': word,
                '词性': flag,
                '标签': row['label']
            })
        # 将当前review的处理结果转换为DataFrame
        temp_data = pd.DataFrame(temp_data_list)
        # 将当前review的DataFrame添加到总的DataFrame中
        processed_data = pd.concat([processed_data, temp_data], ignore_index=True)
    return processed_data

# 构建特征空间和标签
def construct_features(data_frame_recutwordtable):
    """构造特征空间和标签"""
    # 提取每条评论的分词列表
    X = data_frame_recutwordtable.groupby('分词索引')['词语'].apply(list).tolist()
    # 提取每条评论的标签列表，取第一个标签作为单个样本的标签
    Y = data_frame_recutwordtable.groupby('分词索引')['标签'].first().tolist()
    return X, Y

# 训练模型
def train_model(X, Y):
    # 将分词列表转为一维字符串列表，每个元素代表一个评论的分词拼接结果
    X = [' '.join(words) for words in X]
    # 训练集、测试集划分
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    # 词转向量，one-hot编码
    count_vec = CountVectorizer(binary=False, decode_error="replace")
    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)
    print(x_train)
    # 选择最优深度
    # best_depth = 25
    print("正在选择最佳决策树深度 ")
    best_score = 0
    for depth in range(25, 40):
        dtc = tree.DecisionTreeClassifier(max_depth=depth)
        dtc.fit(x_train, y_train)
        score = dtc.score(x_test, y_test)
        print('深度为' + str(depth) + '时准确率：%.5f' % score)
        if score > best_score:
            best_score = score
            best_depth = depth
    print(f"最优决策树深度：{best_depth}")
    # 构建决策树
    print(f"决策树深度：{best_depth}")
    dtc = tree.DecisionTreeClassifier(max_depth=best_depth)
    dtc.fit(x_train, y_train)
    # 决策树可视化
    print("可视化'Decision_Tree_BOW.dot'文件已经保存到目录下")
    export_graphviz(dtc, out_file='Decision_Tree_BOW.dot')
    print("词典已经保存到：'Decision_Tree_BOW_feature.pkl'\n")
    joblib.dump(count_vec.vocabulary_, 'Decision_Tree_BOW_feature.pkl')
    return dtc, x_train, x_test, y_train, y_test, count_vec

# 程序入口
if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    # 加载原始数据
    print("加载原始数据集。。。\n")
    train_data, error_msg = load_data('weibo_senti_100k.csv')
    # 数据预处理
    print("文本清洗、分词、去除标点、停用词处理。。。\n")
    DataFrame_recutwordtable = preprocess_and_clean_data(train_data)
    # 构建特征空间和标签
    print("构建空间和标签。。。\n")
    X, Y = construct_features(DataFrame_recutwordtable)
    # 训练模型
    print("构建决策树模型。。。\n")
    model, x_train, x_test, y_train, y_test, count_vec = train_model(X, Y)
    # 模型评估
    print("模型评估（使用词袋模型）。。。\n")
    y_pred = model.predict(x_test)
    print('在测试集上的准确率：%.2f' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # 模型保存
    print("模型已经保存到：'Decision_Tree_BOW.pkl'\n")
    joblib.dump(model, 'Decision_Tree_BOW.pkl')
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行总时间
    # 将运行时间转换为小时、分钟和秒
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"整个项目的运行时间为：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒。")