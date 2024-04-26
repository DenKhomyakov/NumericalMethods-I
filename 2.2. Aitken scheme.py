import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
valuesX = np.array([2.000, 2.500, 3.000, 3.500, 4.000])
valuesY = np.array([0.010, 1.017, 2.721, 5.149, 8.284])
x = 3.179

def MyFunction(x):
    return np.cos(x + 1) + (x**3 / 8)

# Построение схемы Эйткена
def AitkenScheme(x, valuesX, valuesY):
    n = len(valuesX)
    A = np.zeros((n, n))
    
    for i in range(n):
        A[i, 0] = valuesY[i]
    
    for i in range(1, n):
        for k in range(1, i+1):
            A[i, k] = ((x - valuesX[i-k]) * A[i, k-1] - (x - valuesX[i]) * A[i-1, k-1]) / (valuesX[i] - valuesX[i-k])

    
    print("Матрица, построенная схемой Эйткена:")
    print(A)
    print( )
    
    return A[n-1, n-1]

# Вычисление погрешностей (по Эйткену)
aitkenValue = AitkenScheme(x, valuesX, valuesY)
aitkenTrueValue = MyFunction(x)
aitkenAbsoluteErrorRate = np.abs(aitkenValue - aitkenTrueValue)
aitkenRelativeErrorRate = aitkenAbsoluteErrorRate / np.abs(aitkenTrueValue)

# Вывод результатов (по Эйткену)
print("Результат по Эйткену:")
print("Значение функции y =", aitkenValue)
print("Абсолютная погрешность =", aitkenAbsoluteErrorRate)
print("Относительная погрешность =", aitkenRelativeErrorRate)
  
# Графическое отображение
rangeX = np.linspace(-10, 10, 100)
functionTrueValue = MyFunction(rangeX)
aitkenValue = [AitkenScheme(xi, valuesX, valuesY) for xi in rangeX]
plt.plot(rangeX, functionTrueValue, label ='Истинная функция')
plt.plot(rangeX, aitkenValue, label ='Схема Эйткена')
plt.scatter(valuesX, valuesY, color ='red', label ='Узла интерполирования')
plt.scatter(x, AitkenScheme(x, valuesX, valuesY), color ='green', label ='Точка интерполирования')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Схемой Эйткена')
plt.show()




