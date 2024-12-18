# 此为本人本科毕业设计，仅供大家参考
### 使用SKLearn，WordCloud等库完成
### 具体演示视频可以看我发在bilibili上的视频 P1 大概五分半开始的演示
### [【[毕设] 基于机器学习的新闻评论情感分析方法研究】](https://www.bilibili.com/video/BV1j142197rt/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)
## ⚠⚠⚠ NEWS!
### 新增论文 [基于机器学习的新闻评论情感分析方法研究-论文定稿.pdf](%BB%F9%D3%DA%BB%FA%C6%F7%D1%A7%CF%B0%B5%C4%D0%C2%CE%C5%C6%C0%C2%DB%C7%E9%B8%D0%B7%D6%CE%F6%B7%BD%B7%A8%D1%D0%BE%BF-%C2%DB%CE%C4%B6%A8%B8%E5.pdf)

# 环境配置
### 这里以conda创建虚拟环境为例，如果你是使用conda创建虚拟环境，就从这里开始
```bash
conda create -n ML python=3.12 #这里的ML可以是自定义的环境名
conda activate ML
```
### 如果你使用的不是conda，就从这里开始，如果你使用的是conda，那就继续
### 确保你现在在项目根目录
```bash
pip install -r requirements.txt
```
### 如果出现网络问题，可以尝试使用镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
# 文件说明
### .dot文件为决策树可视化文件，例如：[Decision_Tree_BOW.dot](Decision_Tree_BOW.dot)

### .pkl文件为模型文件，例如：[Decision_Tree_BOW.pkl](Decision_Tree_BOW.pkl)

### 与模型文件同名的.py文件为该模型训练代码，例如：[Decision_Tree_BOW.py](Decision_Tree_BOW.py)

### 训练数据集[weibo_senti_100k.csv](weibo_senti_100k.csv)，来自 https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/weibo_senti_100k ，新浪公司在Github上开源的其旗下新浪微博公司匿名用户评论文本数据集“weibo_senti_100k.csv”并已经对每 条评论进行了情感标注（积极、消极）。数据集共119989条用户评论数据，其中59994条情感标注为“积极的用户评论，和59995条情感标注为“消极”的用户评论。 

### test开头的.csv为验证数据集，使用[spider.py](spider.py)爬取，输入凤凰新闻链接，即可自动创建.csv文件，如果使用[spider.py](spider.py)创建的文件命名后面会加上时间，例如：[test_20240531172628.csv](test_20240531172628.csv)

# 使用方法

### 使用TKInter完成前端编写，运行[demo.py](demo.py)，即可使用，页面包括四个部分，分别是最上方的链接输入框，中间的模型选择下拉栏，中间的新闻评论情感分析预测输出框，下方的每日评论数量变化折线图和词云图。 

### 选择预训练好的模型，再输入test文件，推荐使用[test.csv](test.csv)，或者在线凤凰新网网址都可以，例如：https://sports.ifeng.com/c/8YRoRaeBxRf ,即可在一段时间之后输出最终预测结果


# 学习路径
参考B站 [黑马程序员Python教程，4天快速入门Python数据挖掘，系统精讲+实战案例](https://www.bilibili.com/video/BV1xt411v7z9/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)

参考B站 [黑马程序员3天快速入门python机器学习](https://www.bilibili.com/video/BV1nt411r7tj/?share_source=copy_web)

参考B站 [尚硅谷Python爬虫教程小白零基础速通（含python基础+爬虫案例）](https://www.bilibili.com/video/BV1Db4y1m7Ho/?share_source=copy_web&vd_source=35c49b46a86899e58b3f6414dd834db3)

和ChatGPT的辅助完成，仅供大家参考
