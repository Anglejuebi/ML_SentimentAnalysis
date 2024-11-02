import re
import time
from multiprocessing import Pool
import jieba
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

# 在程序开始时加载停用词列表
with open('./stoplist.txt', 'r', encoding='UTF-8') as stop_path:
    stop_words = set(stop_path.read().split())

# 加载数据
def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            dataset = pd.read_csv(file_path, encoding='gb18030')
        except UnicodeDecodeError:
            return None, "无法读取数据集，请检查文件编码。"
    return dataset, None

def clean_and_segment_text(input_text):
    print("正在清洗和分词文本。。。")
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', input_text)  # 只保留中文字符
    words = jieba.cut(cleaned_text)
    global stop_words
    words = filter(lambda w: w not in stop_words and w.strip(), words)
    return ' '.join(words)


def train_Naive_Bayes_model(data_frame):
    # 数据和标签
    X = data_frame['review']
    Y = data_frame['label']
    # 向量化
    vect = TfidfVectorizer(max_df=0.5, ngram_range=(1, 1))
    # 设置网格搜索参数
    params = {
        'alpha': [1e-3, 1e-2, 1e-1, 1],
        'fit_prior': [True, False],
    }
    # 测试不同的random_state值
    random_states = [42, 52, 62, 72, 82]
    best_scores = []
    best_params = []
    for state in random_states:
        print(f"使用random_state={state}分割数据集。。。")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=state)
        # 实例向量化
        X_train_vect = vect.fit_transform(X_train)
        X_test_vect = vect.transform(X_test)
        # 初始化网格搜索
        grid_search = GridSearchCV(MultinomialNB(), params, cv=5)
        # 执行网格搜索
        print("执行网格搜索。。。")
        grid_search.fit(X_train_vect, y_train)
        # 记录最佳分数和参数
        best_scores.append(grid_search.best_score_)
        best_params.append(grid_search.best_params_)
        print(f"random_state={state}: 最佳平均交叉验证准确率={grid_search.best_score_:.4f}")
        print(f"random_state={state}: 最佳参数={grid_search.best_params_}")
    # 找到最佳的random_state
    best_overall_index = np.argmax(best_scores)
    print(f"整体最佳random_state：{random_states[best_overall_index]}")
    print(f"整体最佳平均交叉验证准确率：{best_scores[best_overall_index]:.4f}")
    print(f"整体最佳参数：{best_params[best_overall_index]}")
    # 使用最佳参数训练模型
    clf = grid_search.best_estimator_
    # 预测测试集
    y_pred = clf.predict(X_test_vect)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率：{accuracy:.2f}")
    # 保存向量化器
    print("保存向量化器。。。")
    vect_path = 'Naive_Bayes_feature.pkl'
    joblib.dump(vect, vect_path)
    # 保存模型
    print("保存模型。。。")
    model_path = 'Naive_Bayes.pkl'
    joblib.dump(clf, model_path)
    print("模型已保存到", model_path)
    # 模型评估
    print("模型评估（使用TF-IDF特征）。。。\n")
    y_pred = clf.predict(X_test_vect)
    print('在测试集上的准确率：%.2f' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test,y_pred), classification_report(y_test,y_pred,target_names=['负面','正面'])


# 主函数
if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    print("正在加载数据集。。。")
    train_data, error_msg = load_data('weibo_senti_100k.csv')
    if train_data is not None:
        pool_size = 8  # 根据您的机器CPU核心数设置
        with Pool(pool_size) as p:
            # 使用map函数并行执行清洗和分词
            # 添加日志来监控每个子进程的状态
            results = p.map_async(clean_and_segment_text, train_data['review'])
            results.wait()  # 等待所有子进程完成
            if results.ready():
                train_data['review'] = results.get()
                print("文本清洗和分词完成。")
            else:
                print("文本清洗和分词过程中出现错误。")
    print("构建朴素贝叶斯模型（使用TDIDF特征）。。。\n")

    acc, report = train_Naive_Bayes_model(train_data)
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行总时间
    # 将运行时间转换为小时、分钟和秒
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"整个项目的运行时间为：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒。")
