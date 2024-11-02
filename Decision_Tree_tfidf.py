import re
import time
from multiprocessing import Pool
import jieba
import joblib
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

def clean_and_segment_text(input_text):
    print("正在清洗和分词文本。。。")
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', input_text)  # 只保留中文字符
    words = jieba.cut(cleaned_text)
    global stop_words
    words = filter(lambda w: w not in stop_words and w.strip(), words)
    return ' '.join(words)

# 训练模型
def train_model(data_frame):
    # 分割文本和标签
    X = data_frame['review']
    Y = data_frame['label']
    # 训练集、测试集划分
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    # 使用TF-IDF进行特征转换
    vect = TfidfVectorizer(max_df=0.5, ngram_range=(1, 1))
    x_train = vect.fit_transform(x_train)
    x_test = vect.transform(x_test)
    print(x_train)
    # # 添加PCA降维
    # from sklearn.decomposition import PCA
    #
    # # 假设我们希望降到的维度为n_components（根据实际需求调整）
    # n_components = 100  # 示例值，根据实际情况调整
    # pca = PCA(n_components=n_components)
    # x_train_pca = pca.fit_transform(x_train.toarray())
    # x_test_pca = pca.transform(x_test.toarray())
    # print(f"PCA降维后的训练集形状: {x_train_pca.shape}, 测试集形状: {x_test_pca.shape}")

    # from sklearn.decomposition import TruncatedSVD
    # # 使用TruncatedSVD进行降维
    # n_components = 100  # 示例值，根据实际情况调整
    # svd = TruncatedSVD(n_components=n_components)
    # x_train_svd = svd.fit_transform(x_train)
    # x_test_svd = svd.transform(x_test)
    # print(f"TruncatedSVD降维后的训练集形状: {x_train_svd.shape}, 测试集形状: {x_test_svd.shape}")

    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    # # 使用LDA进行降维
    # # 使用LDA进行降维之前，将稀疏矩阵转换为密集矩阵
    # x_train_dense = x_train.toarray()
    # x_test_dense = x_test.toarray()
    # n_components = 100
    # lda = LDA(n_components=n_components)
    # x_train_lda = lda.fit_transform(x_train_dense, y_train)
    # x_test_lda = lda.transform(x_test_dense)
    # print(f"LDA降维后的训练集形状: {x_train_lda.shape}, 测试集形状: {x_test_lda.shape}")

    # # 将稀疏矩阵转换为DataFrame
    # tfidf_df = pd.DataFrame(x_train.toarray(), columns=vect.get_feature_names_out())
    # 自动选择最优深度
    best_depth = 28
    # best_score = 0
    # print("正在选择最佳决策树深度 ")
    # for depth in range(1, 61):
    #     dtc = tree.DecisionTreeClassifier(max_depth=depth)
    #     dtc.fit(x_train, y_train)
    #     score = dtc.score(x_test, y_test)
    #     print('深度为' + str(depth) + '时准确率：%.5f' % score)
    #     if score > best_score:
    #         best_score = score
    #         best_depth = depth
    # print(f"最优决策树深度：{best_depth}")
    # 构建决策树
    print(f"决策树深度：{best_depth}")
    dtc = tree.DecisionTreeClassifier(max_depth=best_depth)
    dtc.fit(x_train, y_train)  # 这里改为使用lda处理过的数据
    # 决策树可视化
    print("可视化'Decision_Tree_tfidf.dot'文件已经保存到目录下")
    export_graphviz(dtc, out_file='Decision_Tree_tfidf.dot')
    print("TF-IDF词典已经保存到：'Decision_Tree_tfidf_feature.pkl'\n")
    joblib.dump(vect, 'Decision_Tree_tfidf_feature.pkl')
    return dtc, x_train, x_test, y_train, y_test, vect

# 程序入口
if __name__ == "__main__":
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
    y = train_data['label']

    # 训练模型
    print("构建决策树模型（使用TF-IDF特征）。。。\n")
    model, x_train_svd, x_test_svd, y_train, y_test, tfidf_vec = train_model(train_data)

    # 模型评估
    print("模型评估（使用TF-IDF特征）。。。\n")
    y_pred = model.predict(x_test_svd)
    print('在测试集上的准确率：%.2f' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    #将训练好的模型保存到硬盘
    print("模型已经保存到：'Decision_Tree_tfidf.pkl'\n")
    joblib.dump(model, 'Decision_Tree_tfidf.pkl')
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行总时间
    # 将运行时间转换为小时、分钟和秒
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"整个项目的运行时间为：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒。")