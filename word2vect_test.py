# 导入必要的库
import re
import jieba
import pandas as pd
from gensim.models import Word2Vec

# 加载数据集
def load_dataset(file_path):
    """
    加载数据集函数
    :param file_path: 数据集文件路径
    :return: 数据集中的文本数据
    """
    try:
        dataset = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            dataset = pd.read_csv(file_path, encoding='gb18030')
        except UnicodeDecodeError:
            return None, "无法读取数据集，请检查文件编码。"
    text_data = dataset['review'].tolist()
    return text_data, None


# 文本预处理和分词
def clean_and_segment_text(input_text):
    print("正在清洗和分词文本。。。")
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', input_text)  # 只保留中文字符
    words = jieba.cut(cleaned_text)
    with open('./stoplist.txt', 'r', encoding='UTF-8') as stop_path:
        stop_words = set(stop_path.read().split())
    words = filter(lambda w: w not in stop_words and w.strip(), words)
    return ' '.join(words)


# 构建Word2Vec模型
def build_word2vec_model(sentences, vector_size=100, window=5, min_count=5):
    """
    构建Word2Vec模型函数
    :param sentences: 分词后的文本列表
    :param vector_size: 向量维度大小
    :param window: 上下文窗口大小
    :param min_count: 最小词频阈值
    :return: 训练好的Word2Vec模型
    """
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# 主函数
if __name__ == '__main__':
    train_data, error_msg = load_dataset('weibo_senti_100k.csv')
    if error_msg:
        print(error_msg)
    else:
        # 确保train_data是DataFrame
        train_data = pd.DataFrame(train_data, columns=['review'])
        # 使用apply方法并将结果转换为列表的列表
        sentences = train_data['review'].apply(clean_and_segment_text).tolist()
        # 将每个由空格连接的分词字符串转换为词语列表
        sentences = [sentence.split() for sentence in sentences]
        # 构建Word2Vec模型
        w2v_model = build_word2vec_model(sentences)

        # 保存模型的词向量为txt文件
        w2v_model.wv.save_word2vec_format('word2vec.txt', binary=False)
        print("Word2Vec模型的词向量已保存为txt文件。")

