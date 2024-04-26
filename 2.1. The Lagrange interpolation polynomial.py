import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
valuesX = np.array([2.000, 2.500, 3.000, 3.500, 4.000])
valuesY = np.array([0.010, 1.017, 2.721, 5.149, 8.284])
x = 3.179

def MyFunction(x):
    return np.cos(x + 1) + (x**3 / 8)

# Вычисление многочлена Лагранжа
def LagrangePolynomial(valuesX, valuesY, x):
    n = len(valuesX)
    L = np.zeros(n)
    
    for i in range(n):
        numerator = 1
        denominator = 1
        for k in range(n):
            if k != i:
                numerator *= (x - valuesX[k])
                denominator *= (valuesX[i] - valuesX[k])
        L[i] = numerator / denominator
    
    return np.sum(L * valuesY)

# Вычисление погрешностей
Ay = 0.5 * 10**(-3)
AbsoluteErrorRate = np.abs(LagrangePolynomial(valuesX, valuesY + Ay, x) - MyFunction(x))
RelativeErrorRate = AbsoluteErrorRate / MyFunction(x)

# Вывод результатов
print("Значение функции y =", LagrangePolynomial(valuesX, valuesY, x))
print("Абсолютная погрешность =", AbsoluteErrorRate)
print("Относительная погрешность =", RelativeErrorRate)

# Графическое отображение
rangeX = np.linspace(-10, 15, 100)
functionTrueValue = MyFunction(rangeX)
lagrangeValue = [LagrangePolynomial(valuesX, valuesY, xi) for xi in rangeX]

plt.plot(rangeX, functionTrueValue, label='Истинная функция')
plt.plot(rangeX, lagrangeValue, label='Функция Лагранжа')
plt.scatter(valuesX, valuesY, color='red', label='Узлы интерполирования')
plt.scatter(x, LagrangePolynomial(valuesX, valuesY, x), color='green', label='Точка интерполирования')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяция многочленом Лагранжа')
plt.show()
