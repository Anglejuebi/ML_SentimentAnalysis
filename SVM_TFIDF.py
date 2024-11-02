import re
import time
import jieba
import joblib
import pandas as pd
from sklearn.svm import SVC
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
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

# 训练SVM模型
def train_svm_model(data_frame):
    X = data_frame['review']
    Y = data_frame['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    tfidf_vec = TfidfVectorizer(binary=False, decode_error="replace", min_df=3, max_df=0.9, ngram_range=(1, 2))
    x_train = tfidf_vec.fit_transform(x_train)
    x_test = tfidf_vec.transform(x_test)

    # 定义参数网格，这里仅演示修改核函数，你可以同时调整C和其他相关参数
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}  # 添加gamma参数，这是RBF核的重要参数
    # 使用RBF核函数的SVM分类器进行网格搜索
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)  # 参数为'rbf'
    # 使用训练数据拟合模型
    grid_search.fit(x_train, y_train)
    # 输出每个超参数组合的平均交叉验证准确率
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        print(f"C={cv_results['params'][i]['C']}, Gamma={cv_results['params'][i]['gamma']}: "
              f"平均交叉验证准确率={cv_results['mean_test_score'][i]:.4f}")

    # 输出最佳超参数组合
    best_params = grid_search.best_params_
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    print(f"最优参数C：{best_C}, Gamma：{best_gamma}")

    # 使用最优参数创建非线性核函数（RBF核）的SVM模型
    svm_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
    svm_model.fit(x_train, y_train)
    joblib.dump(tfidf_vec, 'SVM_feature_tfidf.pkl')
    joblib.dump(svm_model, 'SVM_tfidf.pkl')
    y_pred = svm_model.predict(x_test)
    return svm_model, x_train, x_test, y_train, y_test, tfidf_vec, accuracy_score(y_test,y_pred), classification_report(y_test,y_pred,target_names=['负面','正面'])


# 程序入口
if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间
    print("正在加载数据集。。。")
    train_data, error_msg = load_data('weibo_senti_100k.csv')
    if train_data is not None:
        print("分词并过滤停用词。。。")
        # 设置进程池的大小，通常为CPU核心数
        pool_size = 8
        with Pool(pool_size) as p:
            # 使用map函数并行执行清洗和分词
            train_data['review'] = p.map(clean_and_segment_text, train_data['review'])

        print("构建SVM模型（使用TF-IDF特征）。。。\n")
        svm_model, x_train, x_test, y_train, y_test, tfidf_vec, acc, report = train_svm_model(train_data)
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