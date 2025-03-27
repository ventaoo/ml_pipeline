import os
import sys
import pickle
import traceback
import configparser

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from logger import Logger

class classification_models():
    def __init__(self) -> None:
        # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        
        # 初始化日志记录器
        logger = Logger(self.config.getboolean('LOGGER', 'show'))
        self.log = logger.get_logger(__name__)

        # 读取训练和测试数据集
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        # 加载 Word2Vec 预训练模型
        self.word2vec_model = Word2Vec.load(self.config['WORD2VEC']['model_path'])
        
        # 将文本数据转换为向量表示
        self.X_train = np.array([self.text_to_vector(self.word2vec_model, text) for text in self.X_train['question_cleaned']])
        self.X_test = np.array([self.text_to_vector(self.word2vec_model, text) for text in self.X_test['question_cleaned']])
        
        # 转换标签数据格式
        self.y_train = self.y_train.squeeze()
        self.y_test = self.y_test.squeeze()
        
        self.log.info('X converted -> vec')
        
        # 定义项目路径和模型保存路径
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.logistic_regression_path = os.path.join(self.project_path, "logistic_regression.sav")

    def text_to_vector(self, model, text):
        """
        将文本转换为向量表示。
        :param model: 预训练的 Word2Vec 模型
        :param text: 输入的文本字符串
        :return: 计算得到的文本向量
        """
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]  # 仅选择存在于词汇表中的单词
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)  # 计算平均词向量

    def logistic_regression(self, predict=True):
        """
        训练逻辑回归分类器，并进行预测（可选）。
        :param predict: 是否进行测试集预测，默认 True
        :return: 训练好的模型保存状态（是否成功保存）
        """
        classifier = LogisticRegression(class_weight='balanced', max_iter=500)
        
        try:
            classifier.fit(self.X_train, self.y_train)  # 训练逻辑回归模型
        except Exception:
            self.log.error(traceback.format_exc())  # 记录错误日志
            sys.exit(1)  # 发生异常时退出程序

        if predict:
            y_pred = classifier.predict(self.X_test)  # 进行预测
            print(accuracy_score(self.y_test, y_pred))  # 输出准确率

        params = {'path': self.logistic_regression_path}
        return self.save_model(classifier, self.logistic_regression_path, "LOG_REG", params)  # 保存模型

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        """
        保存训练好的模型，并更新配置文件。
        :param classifier: 训练好的分类器模型
        :param path: 模型保存路径
        :param name: 配置文件中的模型名称标识
        :param params: 需要保存到配置文件的参数信息
        :return: 是否成功保存模型（True/False）
        """
        self.config[name] = params  # 更新配置文件
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)  # 使用 pickle 序列化保存模型

        self.log.info(f'{path} is saved')  # 记录日志
        return os.path.isfile(path)  # 检查文件是否成功保存

if __name__ == '__main__':
    model = classification_models()
    model.logistic_regression()