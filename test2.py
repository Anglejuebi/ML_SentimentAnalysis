import tkinter as tk
from tkinter import messagebox
import urllib.request
import json
from datetime import datetime
import os
import pandas as pd
import joblib
import glob
import jieba
import re
# 获取URL的后十一位字符的函数
def get_lasteleven(n_url):
    # 返回URL的最后11个字符
    return n_url[-11:]

# 抓取评论信息的函数
def fetch_comments(url, headers):
    # 构造请求并发送s
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request)
    html = response.read().decode('utf-8')
    # 从响应中提取JSON字符串
    json_str = html.split('=', 1)[1].strip(';')
    # 解析JSON数据
    data = json.loads(json_str)
    return data['comments']

# 将Unix时间戳转换为可读日期和时间的函数
def convert_timestamp(timestamp):
    # 将Unix时间戳转换为UTC时间
    utc_time = datetime.utcfromtimestamp(int(timestamp))
    # 转换为本地时间（根据需要调整时区）
    local_time = utc_time.strftime('%Y-%m-%d %H:%M:%S')
    return local_time

# 检查文件是否存在并返回一个不重名的文件名的函数
def get_unique_filename(filename):
    # 如果文件已存在，则在文件名中添加当前时间戳
    if os.path.exists(filename):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename_without_ext = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        new_filename = f"{filename_without_ext}_{timestamp}{extension}"
        return new_filename
    else:
        return filename

# 获取指定目录下最新的CSV文件
def get_latest_csv(directory_path):
    # 获取目录下所有CSV文件
    list_of_files = glob.glob(os.path.join(directory_path, '*.csv'))
    # 按创建时间排序，获取最新的文件
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# 加载数据
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='gbk')


# 加载模型并预测
def load_model_and_predict(model_path, X_test_vect):
    # 加载模型
    model = joblib.load(model_path)
    # 进行预测
    return model.predict(X_test_vect)

def spider_main(n_url):
    # 构造完整的请求URL
    url = 'https://comment.ifeng.com/get.php?orderby=create_time&docUrl=ucms_' + get_lasteleven(n_url) + '&format=js&job=1&p=1&pageSize=20'
    # 设置请求头weini
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
        'Cookie' : 'userid=1710776144523_vy8wjt1321; prov=cn0791; city=0791; weather_city=jx_nc; sid=A8CC8F3965C4183C0100AAED43B901EE; IF_TIME=1710776362857397; IF_USER=%E5%87%A4%E5%87%B0%E7%BD%91%E5%8F%8BEB78bAP; IF_REAL=1; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2218e524678e79e-0a7efd23795df7-4c657b58-2073600-18e524678e8189d%22%2C%22first_id%22%3A%22%22%2C%22props%22%3A%7B%7D%2C%22%24device_id%22%3A%2218e524678e79e-0a7efd23795df7-4c657b58-2073600-18e524678e8189d%22%7D; region_ip=183.217.29.x; region_ver=1.2'
    }
    # 调用函数抓取评论信息
    comments = fetch_comments(url, headers)
    # 提取'create_time'、'comment_contents'和'uname'，并转换时间戳
    comments_data = [{
        'uname': comment['uname'],
        'create_time': convert_timestamp(comment['create_time']),
        'comment_contents': comment['comment_contents']
    } for comment in comments]
    # 转换为pandas DataFrame
    df = pd.DataFrame(comments_data)
    # 获取不重名的文件名
    unique_filename = get_unique_filename('test.csv')
    # 保存到Excel文件
    df.to_csv(unique_filename, index=False)
    # 输出结果提示
    print(f'评论信息已保存到csv文件：{unique_filename}')

def clean_and_segment_text(text):
    # 定义正则表达式，用于匹配数字和字母
    info = re.compile('[0-9a-zA-Z]')
    # 应用正则表达式，删除数字和字母
    cleaned_text = info.sub('', text)
    # 使用jieba进行分词
    words = jieba.cut(cleaned_text)
    # 读取停用词列表
    with open('./stoplist.txt', 'r', encoding='UTF-8') as stop_path:
        stop_words = stop_path.readlines()
    # 停用词预处理
    stop_words = set(word.strip() for word in stop_words)
    # 去除停用词
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 处理按钮点击的函数
def analyze_sentiment():
    def fetch_and_analyze():
        n_url = url_entry.get()
        if not n_url:
            messagebox.showerror("错误", "请输入有效的 URL。")
            return
        spider_main(n_url)
        # 假设你的CSV文件都在当前目录下
        directory_path = '../'
        latest_csv = get_latest_csv(directory_path)
        # 使用pandas加载最新的CSV文件
        test_data = load_data(latest_csv)
        X_new_test = test_data['comment_contents']
        # 分词处理
        X_new_test_tokenized = X_new_test.apply(clean_and_segment_text)
        # 加载之前保存的向量化器
        # 加载特征词汇表
        # vocabulary = joblib.load('Decision_Tree_feature.pkl')
        # # 使用加载的词汇表创建CountVectorizer实例
        # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocabulary)
        # # 加载模型
        # model = joblib.load('Decision_Tree.pkl')
        vect_path = 'Naive_Bayes_feature.pkl'  # 向量化器文件路径
        vect = joblib.load(vect_path)
        # 使用加载的向量化器转换测试数据
        X_new_test_vect = vect.transform(X_new_test_tokenized)
        # 加载模型并进行情感预测
        model_path = 'Naive_Bayes.pkl'  # 模型文件路径
        new_predictions = load_model_and_predict(model_path, X_new_test_vect)
        # 将预测结果添加到测试数据中
        test_data['predicted_sentiment'] = new_predictions
        test_data['predicted_sentiment'] = test_data['predicted_sentiment'].map({1: '积极', 0: '消极'})
        # 清空文本框内容
        result_text.delete(1.0, tk.END)
        # 在文本框中显示结果
        for index, row in test_data.iterrows():
            result_text.insert(tk.END, f"评论: {row['comment_contents']} - 情感: {row['predicted_sentiment']}\n")


    # 创建一个 Tkinter 窗口
    root = tk.Tk()
    root.title("情感分析")
    # 创建一个标签和一个输入框用于输入 URL
    url_label = tk.Label(root, text="请输入 URL:")
    url_entry = tk.Entry(root, width=150)
    url_label.pack()
    url_entry.pack()
    # 创建一个文本框用于显示情感预测结果
    result_text = tk.Text(root, height=25, width=150)
    result_text.pack()
    # 创建一个按钮来触发情感分析
    analyze_button = tk.Button(root, text="分析情感", command=fetch_and_analyze)
    analyze_button.pack()
    root.mainloop()



if __name__ == '__main__':

    analyze_sentiment()

