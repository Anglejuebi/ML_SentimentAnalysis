import tkinter as tk
from tkinter import scrolledtext
import jieba
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer

# 加载特征词汇表
vocabulary = joblib.load('Decision_Tree_feature.pkl')
# 使用加载的词汇表创建CountVectorizer实例
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocabulary)
# 加载模型
model = joblib.load('Decision_Tree.pkl')

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

# 情感分析的函数
def sentiment_analysis():
    input_text = text_input.get("1.0", tk.END)
    cleaned_text = clean_and_segment_text(input_text)
    print(f'分词结果: {cleaned_text}')  # 控制台输出分词结果
    X = loaded_vec.transform([cleaned_text])
    prediction = model.predict(X)
    result.set(f'输入的语句: "{input_text.strip()}"\n情感分析结果: {"正面" if prediction[0] == 1 else "负面"}')

# 创建主窗口
root = tk.Tk()
root.title("情感分析")

# 创建输入框
text_input = scrolledtext.ScrolledText(root, height=10)
text_input.pack()

# 创建显示结果的标签
result = tk.StringVar()
result_label = tk.Label(root, textvariable=result, height=4)
result_label.pack()

# 创建分析按钮
analyze_button = tk.Button(root, text="分析情感", command=sentiment_analysis)
analyze_button.pack()

# 运行主循环
root.mainloop()
