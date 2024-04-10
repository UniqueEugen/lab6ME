import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Загрузка данных из файла
data = pd.read_csv('winequality-white.csv', sep=';', quotechar='"')
data.drop(['fixed acidity'], axis=1, inplace=True)
data.drop(['residual sugar'], axis=1, inplace=True)
data.drop(['density'], axis=1, inplace=True)


# Визуализация матрицы сходства
plt.figure(figsize=(8, 6))
sns.heatmap(round(abs(data.corr()), 1), annot=True)
plt.title('Проверка на значимость признаков и их корреляцию')
plt.show()

# Разделение данных на обучающую и тестовую выборки
train_input, test_input, train_output, test_output = train_test_split(
    data.drop('quality', axis=1),
    data['quality'],
    test_size=0.2
)

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(train_input, train_output)

# Предсказание на тестовых данных
predictions = model.predict(test_input)

print(predictions)

# Оценка точности модели
accuracy = accuracy_score(test_output, predictions)
print("Точность модели:", accuracy)

# Визуализация матрицы сходства
plt.figure(figsize=(8, 6))
confusion_matrix = sklearn.metrics.confusion_matrix(predictions, test_output)
sns.heatmap(confusion_matrix, annot=True)
plt.title('Матрица сходства')
plt.show()


# Визуализация дерева решений
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=train_input.columns, class_names=[str(i) for i in sorted(data['quality'].unique())], filled=True)
plt.title('Дерево решений')
plt.show()