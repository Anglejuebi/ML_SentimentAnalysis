import re
import time
import jieba
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from multiprocessing import Pool, cpu_count
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

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
    # 仅保留中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', input_text)
    # 使用 jieba 进行分词
    words = jieba.cut(cleaned_text)
    # 过滤停用词和空白字符
    words = [w for w in words if w not in stop_words and w.strip()]
    return ' '.join(words)


def segment_and_vectorize(text, w2v_vectorizer):
    segmented_text = ' '.join(jieba.lcut(text))  # 使用空格连接分词结果，返回字符串
    vectorized_text = w2v_vectorizer.transform([segmented_text])
    # 确保输出是2D且具有一致的特征数量
    if vectorized_text.shape[0] == 0:
        return np.zeros((1, w2v_vectorizer.dim))
    return vectorized_text


# 计算平均词向量
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype='float64')
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector += model[word]
    if nwords:
        feature_vector /= nwords
    return feature_vector


# 将文本转换为平均词向量
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.index_to_key)  # 修改这里
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in
                corpus]
    return np.array(features)


# 获取数据的词向量
def get_word_vectors(data, model):
    words_art = [text.split() for text in data]
    return averaged_word_vectorizer(words_art, model, num_features=100)


# 训练SVM模型
def train_svm_model(train_data, w2v_model):
    X = train_data['cleaned_review']
    Y = train_data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

    # 获取训练和测试数据的词向量
    x_train_w2v = get_word_vectors(x_train, w2v_model)
    x_test_w2v = get_word_vectors(x_test, w2v_model)

    # 定义参数网格，这里添加gamma参数，这是RBF核的重要参数
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}  # 添加gamma参数，这是RBF核的重要参数
    # 使用RBF核函数的SVM分类器进行网格搜索
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)  # 利用所有CPU核心进行并行
    grid_search.fit(x_train_w2v, y_train)

    # 输出每个超参数C和gamma的平均交叉验证准确率
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        print(f"C={cv_results['params'][i]['C']}, Gamma={cv_results['params'][i]['gamma']}: "
              f"平均交叉验证准确率={cv_results['mean_test_score'][i]:.4f}")

    # 输出最佳超参数组合
    best_params = grid_search.best_params_
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    print(f"最优参数C：{best_C}, Gamma：{best_gamma}")

    # 训练SVM模型
    svm_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
    svm_model.fit(x_train_w2v, y_train)
    joblib.dump(svm_model, 'SVM_word2vect.pkl')
    y_pred = svm_model.predict(x_test_w2v)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, target_names=['负面', '正面'])


def parallel_cleaning(data, pool_size):
    with Pool(processes=pool_size) as pool:
        cleaned_reviews = pool.map(clean_and_segment_text, data)
    return cleaned_reviews


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    print("正在加载数据集。。。")
    train_data, error_msg = load_data('weibo_senti_100k.csv')
    if train_data is not None:
        print("正在加载Word2Vec模型。。。")
        w2v_model = KeyedVectors.load_word2vec_format("word2vec.txt", binary=False)

        # 多进程清洗和分词
        start_cleaning = time.time()
        pool_size = cpu_count()  # 使用机器的所有CPU核心
        print(f"使用{pool_size}个进程进行文本清洗和分词。。。")
        train_data['cleaned_review'] = parallel_cleaning(train_data['review'], pool_size)
        end_cleaning = time.time()
        print(f"文本清洗和分词完成，耗时：{end_cleaning - start_cleaning:.2f}秒。")

        print("构建SVM模型（使用word2vec特征）。。。\n")
        acc, report = train_svm_model(train_data, w2v_model)
        print(f'在测试集上的准确率：{acc:.2f}')
        print(report)
    else:
        print(error_msg)
    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算运行总时间
    # 将运行时间转换为小时、分钟和秒
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"整个项目的运行时间为：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒。")
