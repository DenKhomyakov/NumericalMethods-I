import numpy as np
import matplotlib.pyplot as plt

# Входные данные
valuesX = np.array([2.000, 2.500, 3.000, 3.500, 4.000])
valuesY = np.array([0.010, 1.017, 2.721, 5.149, 8.284])

x = 3.179

def MyFunction(x):
    return np.cos(x+1) + (x**3 / 8)

# Линейный сплайн
def LinearSpline(valuesX, valuesY, x):
    index = np.searchsorted(valuesX, x) - 1
    slope = (valuesY[index+1] - valuesY[index]) / (valuesX[index+1] - valuesX[index])

    return valuesY[index] + slope * (x - valuesX[index])

# Кубический сплайн
def CubicSpline(valuesX, valuesY, x):
    index = np.searchsorted(valuesX, x) - 1
    interval = valuesX[index+1] - valuesX[index]
    relativeDistance = (x - valuesX[index]) / interval
    relativeDistanceSquared = relativeDistance * relativeDistance
    relativeDistanceCubed = relativeDistanceSquared * relativeDistance
    blendA = 2 * relativeDistanceCubed - 3 * relativeDistanceSquared + 1
    blendB = -2 * relativeDistanceCubed + 3 * relativeDistanceSquared
    slopeA = relativeDistanceCubed - 2 * relativeDistanceSquared + relativeDistance
    slopeB = relativeDistanceCubed - relativeDistanceSquared
    result = blendA * valuesY[index] + blendB * valuesY[index+1] + slopeA * interval * derivativeValues[index] + slopeB * interval * derivativeValues[index+1]

    return result

# Получение производной заданной функции
def DerivativeMyFunction(x):
    return -np.sin(x+1) + (3*x**2 / 8)

derivativeValues = np.array([DerivativeMyFunction(xi) for xi in valuesX])
print(derivativeValues)

# Эрмитов сплайн
def HermiteSpline(valuesX, valuesY, x):
    index = np.searchsorted(valuesX, x) - 1
    interval = valuesX[index+1] - valuesX[index]
    relativePosition = (x - valuesX[index]) / interval
    startBlend = 2*relativePosition**3 - 3*relativePosition**2 + 1
    endBlend = -2*relativePosition**3 + 3*relativePosition**2
    startSlope = relativePosition**3 - 2*relativePosition**2 + relativePosition
    endSlope = relativePosition**3 - relativePosition**2

    return startBlend*valuesY[index] + endBlend*valuesY[index+1] + startSlope*interval*derivativeValues[index] + endSlope*interval*derivativeValues[index+1]


# Точки интерполяции
rangeX= np.linspace(2, 4, 100)

linearInterpolation = np.array([LinearSpline(valuesX, valuesY, xi) for xi in rangeX])
cubicInterpolation = np.array([CubicSpline(valuesX, valuesY, xi) for xi in rangeX])
hermiteInterpolation = np.array([HermiteSpline(valuesX, valuesY, xi) for xi in rangeX])

linearInterpolation = LinearSpline(valuesX, valuesY, rangeX)
cubicInterpolation = CubicSpline(valuesX, valuesY, rangeX)
hermiteInterpolation = HermiteSpline(valuesX, valuesY, rangeX)


trueValue = MyFunction(x)
functionTrueValue = MyFunction(rangeX)

# Построение график линейного сплайна
plt.plot(rangeX, functionTrueValue, label='Истинная функция', color='black')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.plot(rangeX, linearInterpolation, label='Линейный сплайн', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линейный сплайн')
plt.show()

# Построение графика кубического сплайна
plt.plot(rangeX, functionTrueValue, label='Истинная функция', color='black')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.plot(rangeX, cubicInterpolation, label='Кубический сплайн', color='green')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Кубический сплайн')
plt.show()

# Построение графика эрмитового сплайна
plt.plot(rangeX, functionTrueValue, label='Истинная функция', color='black')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.plot(rangeX, hermiteInterpolation, label='Эрмитов сплайн', color='orange')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Эрмитов сплайн')
plt.show()

# Построение графика
plt.plot(rangeX, functionTrueValue, label='Истинная функция', color='black')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.plot(rangeX, linearInterpolation, label='Линейный сплайн', color='blue')
plt.plot(rangeX, cubicInterpolation, label='Кубический сплайн', color='green')
plt.plot(rangeX, hermiteInterpolation, label='Эрмитов сплайн', color='orange')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сплайны')
plt.show()
