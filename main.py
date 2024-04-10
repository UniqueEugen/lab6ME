import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Загрузка данных из CSV-файла
dataset = pd.read_csv('Mall_Customers.csv')
dataset= dataset[:100]

label_encoder = LabelEncoder()

# Замена строковых значений на числовые значения в столбце "Genre"
dataset['Genre'] = label_encoder.fit_transform(dataset['Genre'])

# Удаление ненужных столбцов
dataset.drop(['CustomerID'], axis=1, inplace=True)


# Визуализация матрицы сходства
plt.figure(figsize=(8, 6))
sns.heatmap(round(abs(dataset.corr()), 1), annot=True)
plt.title('Проверка на значимость признаков и их корреляцию')
plt.show()

# Разделение данных на обучающую и тестовую выборки
train_input, test_input, train_output, test_output = train_test_split(
    dataset.drop('Genre', axis=1),
    dataset['Genre'],
    test_size=0.2
)

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(train_input, train_output)

# Предсказание на тестовых данных
predictions = model.predict(test_input)

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
plot_tree(model, feature_names=train_input.columns, class_names=['0', '1'], filled=True)
plt.title('Дерево решений')
plt.show()

#В визуализации дерева решений, каждый блок представляет узел дерева и содержит информацию о разделении данных на основе определенного признака. В каждом блоке обычно отображаются следующие элементы:
# разделения: Описывает условие, по которому происходит разделение данных. Например, "Age <= 30" означает, что данные разделяются на две ветви в зависимости от значения возраста, где значения меньше или равные 30 переходят влево, а большие значения переходят вправо.
#Gini-индекс: Показывает меру неопределенности разделения в данном узле. Чем ближе значение Gini-индекса к 0, тем чище разделение, что означает, что классы в каждой ветви становятся более однородными.
#Образцы: Показывает количество образцов, попадающих в данный узел.
#Классы: Показывает распределение классов образцов в данном узле. Например, в узле может быть 30 образцов класса 0 и 20 образцов класса 1.
#Value: Показывает распределение классов образцов в данном узле в виде массива. Например, [30, 20] означает, что в узле находится 30 образцов класса 0 и 20 образцов класса 1.
#Class: Показывает класс, который будет назначен данным узлом в случае принятия решения. Например, в узле может быть указан класс 0 или класс 1.
#Это основные элементы, которые обычно присутствуют в блоках визуализации дерева решений. Они помогают понять, как модель принимает решения на основе признаков данных.