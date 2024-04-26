import numpy as np
import matplotlib.pyplot as plt

valuesX = np.array([2.000, 2.500, 3.000, 3.500, 4.000])
valuesY = np.array([0.010, 1.017, 2.721, 5.149, 8.284])

def MyFunction(x):
    return np.cos(x + 1) + (x**3 / 8)

def GaussForward(valuesX, valuesY, printMatrix=False):
    n = len(valuesX)
    step = valuesX[1] - valuesX[0]

    # Создаем матрицу разделенных разностей и заполняем первый столбец значениями y
    matrix = np.zeros((n, n))
    matrix[:, 0] = valuesY

    # Вычисляем разделенные разности
    for j in range(1, n):
        for i in range(n - j):
            # Рекурсивное вычисление разделенных разностей
            matrix[i, j] = matrix[i + 1, j - 1] - matrix[i, j - 1]

    # Вывод матрицы разделенных разностей
    if printMatrix:
        print("Матрица разделенных разностей:")
        for i in range(n):
            for j in range(n - i):
                print(matrix[i, j], end="\t")
            print()

    # Получаем значение первой производной
    firstDerivative = matrix[0, 1] / step

    return firstDerivative

# Вычисляем приблизительное и точное значение первой производной
approximateValue = GaussForward(valuesX, valuesY, printMatrix=True)
exactValue = -np.sin(3.179 + 1) + (3.179**2 / 8)

# Вычисляем абсолютную погрешность
absoluteErrorRate = abs(approximateValue - exactValue)

print("Приблизительное значение первой производной:", approximateValue)
print("Точное значение первой производной:", exactValue)
print("Абсолютная погрешность:", absoluteErrorRate)

# Построение графика основной функции и графика интерполяции
#rangeX = np.linspace(-10, 15, 100)
#functionTrueValue = -np.sin(rangeX + 1) + (rangeX**2 / 8)
#gaussValue = [GaussForward(valuesX, valuesY) for xi in rangeX]

#plt.plot(rangeX, functionTrueValue, label='Истинная функция', color='orange')
#plt.plot(rangeX, gaussValue, label='Функция Гаусса', color='blue')
#plt.scatter(valuesX, valuesY, color='red', label='Узлы интерполирования')
#plt.scatter(3.179, GaussForward(valuesX, valuesY), color='green', label='Точка интерполирования')
#plt.legend()

#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Вычисление значения первой производной в заданной точке')
#plt.show()
