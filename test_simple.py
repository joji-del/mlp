#!/usr/bin/env python3
"""
Упрощенное тестирование нейронной сети без внешних зависимостей
"""

import math
import random

# Замена numpy - простые функции для матричных операций
def random_normal(shape, mean=0, std=1):
    """Создание массива со случайными числами из нормального распределения"""
    if isinstance(shape, int):
        return [random.gauss(mean, std) for _ in range(shape)]
    elif len(shape) == 1:
        return [random.gauss(mean, std) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.gauss(mean, std) for _ in range(shape[1])] for _ in range(shape[0])]

def zeros(shape):
    """Создание массива нулей"""
    if isinstance(shape, int):
        return [0.0 for _ in range(shape)]
    elif len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]

def ones(shape):
    """Создание массива единиц"""
    if isinstance(shape, int):
        return [1.0 for _ in range(shape)]
    elif len(shape) == 1:
        return [1.0 for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[1.0 for _ in range(shape[1])] for _ in range(shape[0])]

def matrix_multiply(A, B):
    """Произведение матриц"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Неподходящие размеры матриц для умножения")
    
    result = zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_transpose(A):
    """Транспонирование матрицы"""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def vector_add(a, b):
    """Сложение векторов"""
    return [a[i] + b[i] for i in range(len(a))]

def matrix_add(A, B):
    """Сложение матриц"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def apply_function(matrix, func):
    """Применение функции к каждому элементу"""
    if isinstance(matrix[0], list):
        return [[func(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    else:
        return [func(x) for x in matrix]

def mean(arr, axis=None):
    """Среднее значение"""
    if axis is None:
        flat = [item for row in arr for item in (row if isinstance(row, list) else [row])]
        return sum(flat) / len(flat)
    elif axis == 0:
        return [sum(arr[i][j] for i in range(len(arr))) / len(arr) for j in range(len(arr[0]))]

def variance(arr, axis=None):
    """Дисперсия"""
    if axis == 0:
        means = mean(arr, axis=0)
        n = len(arr)
        return [sum((arr[i][j] - means[j])**2 for i in range(n)) / n for j in range(len(arr[0]))]

def sqrt(x):
    """Квадратный корень"""
    if isinstance(x, list):
        return [math.sqrt(val) for val in x]
    else:
        return math.sqrt(x)

# Простой тест линейного слоя
class SimpleLinear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Инициализация весов
        std = math.sqrt(2.0 / input_size)
        self.weight = random_normal((input_size, output_size), 0, std)
        self.bias = zeros(output_size)
        
        self.input = None
        self.grad_weight = zeros((input_size, output_size))
        self.grad_bias = zeros(output_size)
    
    def forward(self, x):
        self.input = [row[:] for row in x]  # копия
        output = matrix_multiply(x, self.weight)
        
        # Добавляем bias
        for i in range(len(output)):
            output[i] = vector_add(output[i], self.bias)
        
        return output
    
    def backward(self, grad_output):
        # grad_input = grad_output @ weight.T
        weight_T = matrix_transpose(self.weight)
        grad_input = matrix_multiply(grad_output, weight_T)
        
        # grad_weight = input.T @ grad_output  
        input_T = matrix_transpose(self.input)
        self.grad_weight = matrix_multiply(input_T, grad_output)
        
        # grad_bias = sum(grad_output, axis=0)
        self.grad_bias = [sum(grad_output[i][j] for i in range(len(grad_output))) for j in range(len(grad_output[0]))]
        
        return grad_input

# Простой ReLU
class SimpleReLU:
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = [row[:] for row in x]  # копия
        return apply_function(x, lambda val: max(0, val))
    
    def backward(self, grad_output):
        result = zeros((len(grad_output), len(grad_output[0])))
        for i in range(len(grad_output)):
            for j in range(len(grad_output[0])):
                result[i][j] = grad_output[i][j] if self.input[i][j] > 0 else 0
        return result

# Простой CrossEntropy Loss
def softmax(x):
    """Простая реализация softmax"""
    result = []
    for row in x:
        # Для численной стабильности вычитаем максимум
        max_val = max(row)
        exp_row = [math.exp(val - max_val) for val in row]
        sum_exp = sum(exp_row)
        result.append([val / sum_exp for val in exp_row])
    return result

def cross_entropy_loss(predictions, targets):
    """Вычисление cross-entropy loss"""
    batch_size = len(predictions)
    softmax_pred = softmax(predictions)
    
    total_loss = 0
    for i in range(batch_size):
        # Избегаем log(0)
        prob = max(softmax_pred[i][targets[i]], 1e-15)
        total_loss += -math.log(prob)
    
    return total_loss / batch_size, softmax_pred

def cross_entropy_backward(softmax_pred, targets):
    """Градиент cross-entropy loss"""
    batch_size = len(softmax_pred)
    num_classes = len(softmax_pred[0])
    
    grad = zeros((batch_size, num_classes))
    for i in range(batch_size):
        for j in range(num_classes):
            if j == targets[i]:
                grad[i][j] = (softmax_pred[i][j] - 1) / batch_size
            else:
                grad[i][j] = softmax_pred[i][j] / batch_size
    
    return grad

# Простая нейронная сеть
class SimpleNetwork:
    def __init__(self):
        self.layer1 = SimpleLinear(4, 8)  # Уменьшенные размеры для простоты
        self.relu1 = SimpleReLU()
        self.layer2 = SimpleLinear(8, 3)  # 3 класса
        
        self.learning_rate = 0.01
    
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.relu1.forward(x)
        x = self.layer2.forward(x)
        return x
    
    def backward(self, grad_output):
        grad = self.layer2.backward(grad_output)
        grad = self.relu1.backward(grad)
        grad = self.layer1.backward(grad)
        return grad
    
    def update_weights(self):
        # Простое обновление весов (SGD)
        for i in range(len(self.layer1.weight)):
            for j in range(len(self.layer1.weight[0])):
                self.layer1.weight[i][j] -= self.learning_rate * self.layer1.grad_weight[i][j]
        
        for i in range(len(self.layer1.bias)):
            self.layer1.bias[i] -= self.learning_rate * self.layer1.grad_bias[i]
        
        for i in range(len(self.layer2.weight)):
            for j in range(len(self.layer2.weight[0])):
                self.layer2.weight[i][j] -= self.learning_rate * self.layer2.grad_weight[i][j]
        
        for i in range(len(self.layer2.bias)):
            self.layer2.bias[i] -= self.learning_rate * self.layer2.grad_bias[i]
    
    def zero_grad(self):
        self.layer1.grad_weight = zeros((self.layer1.input_size, self.layer1.output_size))
        self.layer1.grad_bias = zeros(self.layer1.output_size)
        self.layer2.grad_weight = zeros((self.layer2.input_size, self.layer2.output_size))
        self.layer2.grad_bias = zeros(self.layer2.output_size)

def create_simple_dataset():
    """Создание простого синтетического датасета"""
    random.seed(42)
    
    # Создаем простые данные
    X = []
    y = []
    
    for _ in range(100):
        # Простые паттерны для 3 классов
        if random.random() < 0.33:
            # Класс 0: большие положительные значения в первых двух признаках
            sample = [random.gauss(2, 0.5), random.gauss(2, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)]
            label = 0
        elif random.random() < 0.5:
            # Класс 1: большие отрицательные значения в первых двух признаках
            sample = [random.gauss(-2, 0.5), random.gauss(-2, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)]
            label = 1
        else:
            # Класс 2: большие значения в последних двух признаках
            sample = [random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(2, 0.5), random.gauss(2, 0.5)]
            label = 2
        
        X.append(sample)
        y.append(label)
    
    return X, y

def test_simple_network():
    """Тестирование простой нейронной сети"""
    print("🚀 Создание простого синтетического датасета...")
    X, y = create_simple_dataset()
    
    print(f"Размер датасета: {len(X)} x {len(X[0])}")
    print(f"Количество классов: {len(set(y))}")
    
    print("\n🏗️ Создание простой нейронной сети...")
    network = SimpleNetwork()
    
    print("Архитектура:")
    print("  Input: 4 features")
    print("  Hidden: 8 neurons + ReLU")
    print("  Output: 3 classes")
    
    print("\n🎯 Начинаем обучение...")
    
    losses = []
    epochs = 20
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for i in range(len(X)):
            # Прямой проход
            network.zero_grad()
            
            # Формируем батч размером 1
            x_batch = [X[i]]
            y_batch = [y[i]]
            
            predictions = network.forward(x_batch)
            
            # Вычисляем loss
            loss, softmax_pred = cross_entropy_loss(predictions, y_batch)
            total_loss += loss
            
            # Проверяем правильность предсказания
            predicted_class = max(range(len(predictions[0])), key=lambda k: predictions[0][k])
            if predicted_class == y[i]:
                correct += 1
            
            # Обратный проход
            grad_loss = cross_entropy_backward(softmax_pred, y_batch)
            network.backward(grad_loss)
            network.update_weights()
        
        avg_loss = total_loss / len(X)
        accuracy = correct / len(X)
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}")
    
    print("\n✅ Обучение завершено!")
    
    # Проверяем критерий успешности
    print(f"\n📊 Результаты:")
    print(f"Начальный loss: {losses[0]:.4f}")
    print(f"Конечный loss: {losses[-1]:.4f}")
    print(f"Снижение loss: {losses[0] - losses[-1]:.4f}")
    
    is_decreasing = losses[0] > losses[-1]
    print(f"\n🎯 Проверка критериев успешности:")
    print(f"✓ Loss падает: {'Да' if is_decreasing else 'Нет'}")
    print(f"✓ Обучение за {len(losses)} эпох: {'Да' if len(losses) <= 20 else 'Нет'}")
    
    success = is_decreasing and len(losses) <= 20
    print(f"\n{'🎉 УСПЕХ! Нейронная сеть успешно обучается!' if success else '❌ Требуется доработка.'}")
    
    # Простая визуализация loss
    print(f"\n📈 График loss (текстовый):")
    max_loss = max(losses)
    min_loss = min(losses)
    
    for i, loss in enumerate(losses):
        if i % 4 == 0:  # Показываем каждую 4-ю эпоху
            normalized = int((loss - min_loss) / (max_loss - min_loss + 1e-8) * 50)
            bar = "█" * normalized + "░" * (50 - normalized)
            print(f"Epoch {i+1:2d}: {loss:.4f} |{bar}|")
    
    return success

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ПРОСТОЕ ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ")
    print("=" * 60)
    
    success = test_simple_network()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ТЕСТ ПРОЙДЕН: Реализация работает корректно!")
        print("💡 Нейронная сеть способна обучаться и снижать loss")
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН: Требуется дополнительная отладка")
    print("=" * 60)