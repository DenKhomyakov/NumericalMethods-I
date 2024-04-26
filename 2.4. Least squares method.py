import numpy as np
import matplotlib.pyplot as plt

valuesX = np.array([2.000, 2.500, 3.000, 3.500, 4.000])
valuesY = np.array([0.010, 1.017, 2.721, 5.149, 8.284])
interpolationX = 3.179

n = valuesX.size
m = 1

def MyFunction(x):
    return np.cos(x + 1) + (x ** 3 / 8)

flag = 0

# Алгебраический полином
def PowerPreparation(i, x):
    return x ** i

# Полином Чебышёва
def ChebyshevPolynomial(k, x):
    result = 1

    if k > 0:
        buffer = result
        result = x

    for i in range(2, k + 1):
        tmp = 2 * x * result - buffer
        buffer = result
        result = tmp

    return result

# Полином Лежандра
def LegendrePolynomial(k, x):
    result = 1

    if k > 0:
        buffer = result
        result = x

    for i in range(2, k + 1):
        tmp = (1 / (i)) * ((2 * i - 1) * x * result + (i + 1) * buffer)
        buffer = result
        result = tmp

    return result

# Полином Чебышёва с дискретной переменной
def ChebyshevPolynomialDiscreteVariable(k, x):
    result = 1

    if k > 0:
        amountValuesX = np.sum(valuesX)
        buffer = result
        result = x - (1 / n) * amountValuesX

    for i in range(2, k + 1):
        amountValuesX = bufferAmountValuesX = amountSquaresValuesX = 0

        for j in range(n):
            amountValuesX += valuesX[j] * (ChebyshevPolynomialDiscreteVariable(i - 1, valuesX[j]) ** 2)
            bufferAmountValuesX += ChebyshevPolynomialDiscreteVariable(i - 1, valuesX[j]) ** 2
            amountSquaresValuesX += ChebyshevPolynomialDiscreteVariable(i - 2, valuesX[j]) ** 2

        tmp = (x - amountValuesX / bufferAmountValuesX) * result - (bufferAmountValuesX / amountSquaresValuesX) * buffer
        buffer = result
        result = tmp

    return result

# Алгебраический полином
def AlgebraicPolynomial(PowerPreparation, x):
    algebraicMatrix = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(n):
                algebraicMatrix[i][j] += PowerPreparation(i, valuesX[k]) * PowerPreparation(j, valuesX[k])

    if flag == 1:
        print(algebraicMatrix)

    rightHandSideVector = np.zeros(m + 1)

    for i in range(m + 1):
        for k in range(n):
            rightHandSideVector[i] += PowerPreparation(i, valuesX[k]) * valuesY[k]

    coefficientsVector = np.linalg.solve(algebraicMatrix, rightHandSideVector)
    result = 0

    for i in range(m + 1):
        result += coefficientsVector[i] * PowerPreparation(i, x)

    if flag == 0:
        return result

# Ортогональный полином
def OrthogonalPolynomial(phi, x):
    transformenValuesX = np.zeros(n)

    for i in range(n):
        transformenValuesX[i] = 2 * ((valuesX[i] - valuesX[0]) / (valuesX[n - 1] - valuesX[0])) - 1

    orthogonalMatrix = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(n):
                orthogonalMatrix[i][j] += phi(i, transformenValuesX[k]) * phi(j, transformenValuesX[k])

    if flag == 1:
        print(orthogonalMatrix)

    rightHandSideVector = np.zeros(m + 1)

    for i in range(m + 1):
        for k in range(n):
            rightHandSideVector[i] += phi(i, transformenValuesX[k]) * valuesY[k]

    coefficientsVector = np.linalg.solve(orthogonalMatrix, rightHandSideVector)

    x = 2 * ((x - valuesX[0]) / (valuesX[n - 1] - valuesX[0])) - 1
    result = 0

    for i in range(m + 1):
        result += coefficientsVector[i] * phi(i, x)

    if flag == 0:
        return result

# Матрица Грамма
flag = 1
print("Алгебраический полином:")
AlgebraicPolynomial(PowerPreparation, interpolationX)
print("Полином Чебышёва:")
OrthogonalPolynomial(ChebyshevPolynomial, interpolationX)
print("Полином Лежандра:")
OrthogonalPolynomial(LegendrePolynomial, interpolationX)
print("Полином Чебышёва дискретной переменной:")
AlgebraicPolynomial(ChebyshevPolynomialDiscreteVariable, interpolationX)

# График "Алгебраический полином"
flag = 0

newX = np.linspace(np.min(valuesX), np.max(valuesX), 100)
newY = [AlgebraicPolynomial(PowerPreparation, i) for i in newX]
valuesXForPlotting = np.arange(np.min(valuesX), np.max(valuesX), 0.01)

plt.plot(newX, newY, label='Алгебраический полином', color='black')
#plt.plot(valuesXForPlotting, MyFunction(valuesXForPlotting), label='y=f(x)', color='orange')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# График "Полином Чебышёва"
flag = 0

newX = np.linspace(np.min(valuesX), np.max(valuesX), 100)
newY = [OrthogonalPolynomial(ChebyshevPolynomial, i) for i in newX]
valuesXForPlotting = np.arange(np.min(valuesX), np.max(valuesX), 0.01)

plt.plot(newX, newY, label='Полином Чебышёва', color='black')
#plt.plot(valuesXForPlotting, MyFunction(valuesXForPlotting), label='y=f(x)', color='orange')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# График "Полином Лежандра"
flag = 0

newX = np.linspace(np.min(valuesX), np.max(valuesX), 100)
newY = [OrthogonalPolynomial(LegendrePolynomial, i) for i in newX]
valuesXForPlotting = np.arange(np.min(valuesX), np.max(valuesX), 0.01)

plt.plot(newX, newY, label='Полином Лежандра', color='black')
#plt.plot(valuesXForPlotting, MyFunction(valuesXForPlotting), label='y=f(x)', color='orange')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# График "Полином Чебышёва дискретной переменной"
flag = 0

newX = np.linspace(np.min(valuesX), np.max(valuesX), 100)
newY = [AlgebraicPolynomial(ChebyshevPolynomialDiscreteVariable, i) for i in newX]
valuesXForPlotting = np.arange(np.min(valuesX), np.max(valuesX), 0.01)

plt.plot(newX, newY, label='Полином Чебышёва дискретной переменной', color='black')
#plt.plot(valuesXForPlotting, MyFunction(valuesXForPlotting), label='y=f(x)', color='orange')
plt.scatter(valuesX, valuesY, label='(xi, yi)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
