#!/usr/bin/env python3
"""
Тестирование исправленной нейронной сети
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

# Установим seed для воспроизводимости
np.random.seed(42)

# ========== БАЗОВЫЕ КЛАССЫ ==========

class Layer:
    """
    Базовый класс для всех слоев нейронной сети
    """
    def __init__(self):
        self.training = True

    def forward(self, x):
        """
        Прямое распространение
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Обратное распространение
        """
        raise NotImplementedError

    def train(self):
        """
        Переключение в режим обучения
        """
        self.training = True

    def eval(self):
        """
        Переключение в режим инференса
        """
        self.training = False

    def __call__(self, x):
        return self.forward(x)


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        output = np.maximum(0, x)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = 1/(1+ np.exp(-x))
        return self.output

    def backward(self, grad_output):
        sigmoid_derivative = self.output * (1 - self.output)
        grad_input = sigmoid_derivative * grad_output
        return grad_input


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        self.output = np.clip(self.output, -0.999999, 0.999999)
        return self.output

    def backward(self, grad_output):
        tanh_derivative = 1 - self.output**2
        grad_input = tanh_derivative * grad_output
        return grad_input


class Linear(Layer):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        # Инициализация весов (Kaiming)
        std = np.sqrt(2.0 / input_size)
        self.weight = np.random.randn(input_size, output_size) * std

        # Инициализация bias
        if self.use_bias:
            self.bias = np.zeros(output_size)
        else:
            self.bias = None

        # Переменные для сохранения входных данных и градиентов
        self.input = None
        self.grad_weight = np.zeros_like(self.weight)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)
        else:
            self.grad_bias = None

    def forward(self, x):
        self.input = x.copy()
        output = x @ self.weight

        if self.use_bias:
            output += self.bias

        return output

    def backward(self, grad_output):
        grad_input = grad_output @ self.weight.T
        self.grad_weight += self.input.T @ grad_output

        if self.use_bias:
            self.grad_bias += np.sum(grad_output, axis=0)

        return grad_input

    def update_weights(self, learning_rate=0.01):
        if self.grad_weight is not None:
            self.weight -= learning_rate * self.grad_weight

        if self.use_bias and self.grad_bias is not None:
            self.bias -= learning_rate * self.grad_bias

    def zero_grad(self):
        self.grad_weight = np.zeros_like(self.grad_weight)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.grad_bias)

    def parameters(self):
        if self.bias is not None:
            return (self.weight, self.bias)
        else:
            return (self.weight,)


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self.layer_outputs = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        self.layer_outputs = []
        self.layer_outputs.append(x)
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            self.layer_outputs.append(output)
        return output

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, learning_rate=0.01):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(learning_rate)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

    def train(self):
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        super().eval()
        for layer in self.layers:
            layer.eval()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params = layer.parameters()
                if isinstance(params, tuple):
                    yield from params
                else:
                    yield params


class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
            output = x * self.mask
        else:
            output = x
            self.mask = None
        return output

    def backward(self, grad_output):
        if self.training:
            grad_input = grad_output * self.mask
        else:
            grad_input = grad_output
        return grad_input


class BatchNorm(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Обучаемые параметры gamma и beta
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Накопленная статистика
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Переменные для backward pass
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.input = None
        self.grad_gamma = np.zeros(num_features)
        self.grad_beta = np.zeros(num_features)

    def forward(self, x):
        self.input = x

        if self.training:
            # Вычисляем статистику текущего batch
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)

            # Обновляем накопленную статистику (ИСПРАВЛЕНО)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            # Используем накопленную статистику
            mean = self.running_mean
            var = self.running_var

        # Нормализация
        self.normalized = (x - mean) / np.sqrt(var + self.eps)

        # Масштабирование и сдвиг
        output = self.gamma * self.normalized + self.beta

        return output

    def backward(self, grad_output):
        n = self.input.shape[0]
        # Вычисляем градиенты по параметрам
        self.grad_gamma = np.sum(grad_output * self.normalized, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        grad_normalized = grad_output * self.gamma

        # Вычисляем градиент по входу
        grad_var = np.sum(grad_normalized * (self.input - self.batch_mean), axis=0) * (-0.5) * (self.batch_var + self.eps)**(-1.5)

        grad_mean = np.sum(grad_normalized, axis=0) * (-1) / np.sqrt(self.batch_var + self.eps) + grad_var * np.mean(-2 * (self.input - self.batch_mean), axis=0)

        grad_input = (grad_normalized / np.sqrt(self.batch_var + self.eps)) + grad_var * 2 * (self.input - self.batch_mean) / n + grad_mean / n

        return grad_input

    def update_weights(self, learning_rate=0.01):
        if self.grad_gamma is not None:
            self.gamma -= learning_rate * self.grad_gamma

        if self.grad_beta is not None:
            self.beta -= learning_rate * self.grad_beta

    def zero_grad(self):
        self.grad_gamma = np.zeros(self.num_features)
        self.grad_beta = np.zeros(self.num_features)


# ========== ФУНКЦИИ ПОТЕРЬ ==========

def softmax(x):
    """Устойчивая реализация softmax"""
    x_stable = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot_encode(labels, num_classes):
    """Преобразование меток в one-hot кодировку"""
    batch_size = labels.shape[0]
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot


class CrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        # Применяем softmax к предсказаниям
        self.softmax_pred = softmax(predictions)

        # Вычисляем cross-entropy loss
        batch_size = predictions.shape[0]
        correct_class_probs = self.softmax_pred[np.arange(batch_size), targets]
        loss = -np.mean(np.log(correct_class_probs + 1e-15))

        return loss

    def backward(self):
        batch_size = self.predictions.shape[0]
        num_classes = self.predictions.shape[1]

        # Создаем one-hot кодировку для целей
        one_hot_targets = one_hot_encode(self.targets, num_classes)

        # Градиент: (softmax_pred - one_hot_targets) / batch_size
        grad = (self.softmax_pred - one_hot_targets) / batch_size
        return grad


# ========== ОПТИМИЗАТОР ==========

class AdamFixed:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Словари для хранения моментов для каждого параметра каждого слоя
        self.m = {}  # first moment
        self.v = {}  # second moment
        self._layer_steps = {}  # time steps for each layer

    def update(self, layer, layer_id):
        """Обновление параметров слоя с помощью Adam"""
        # Увеличиваем счетчик шагов для данного слоя
        if layer_id not in self._layer_steps:
            self._layer_steps[layer_id] = 0
        
        self._layer_steps[layer_id] += 1
        t = self._layer_steps[layer_id]

        # Обновляем веса, если есть градиенты
        if hasattr(layer, 'grad_weight') and layer.grad_weight is not None:
            grad_w = layer.grad_weight
            
            # Инициализируем моменты для весов
            weight_key = f"{layer_id}_weight"
            if weight_key not in self.m:
                self.m[weight_key] = np.zeros_like(layer.weight)
                self.v[weight_key] = np.zeros_like(layer.weight)

            # Обновляем первый момент (momentum)
            self.m[weight_key] = self.beta1 * self.m[weight_key] + (1 - self.beta1) * grad_w

            # Обновляем второй момент (RMSprop)
            self.v[weight_key] = self.beta2 * self.v[weight_key] + (1 - self.beta2) * (grad_w ** 2)

            # Коррекция смещения
            m_corrected = self.m[weight_key] / (1 - self.beta1 ** t)
            v_corrected = self.v[weight_key] / (1 - self.beta2 ** t)

            # Обновляем веса
            layer.weight -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.eps)

        # Обновляем bias аналогично весам
        if hasattr(layer, 'grad_bias') and layer.grad_bias is not None:
            grad_b = layer.grad_bias
            
            bias_key = f"{layer_id}_bias"
            if bias_key not in self.m:
                self.m[bias_key] = np.zeros_like(layer.bias)
                self.v[bias_key] = np.zeros_like(layer.bias)

            self.m[bias_key] = self.beta1 * self.m[bias_key] + (1 - self.beta1) * grad_b
            self.v[bias_key] = self.beta2 * self.v[bias_key] + (1 - self.beta2) * (grad_b ** 2)

            m_corrected_b = self.m[bias_key] / (1 - self.beta1 ** t)
            v_corrected_b = self.v[bias_key] / (1 - self.beta2 ** t)

            layer.bias -= self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.eps)

        # Обновляем gamma и beta для BatchNorm
        if hasattr(layer, 'grad_gamma') and layer.grad_gamma is not None:
            grad_gamma = layer.grad_gamma
            
            gamma_key = f"{layer_id}_gamma"
            if gamma_key not in self.m:
                self.m[gamma_key] = np.zeros_like(layer.gamma)
                self.v[gamma_key] = np.zeros_like(layer.gamma)

            self.m[gamma_key] = self.beta1 * self.m[gamma_key] + (1 - self.beta1) * grad_gamma
            self.v[gamma_key] = self.beta2 * self.v[gamma_key] + (1 - self.beta2) * (grad_gamma ** 2)

            m_corrected_gamma = self.m[gamma_key] / (1 - self.beta1 ** t)
            v_corrected_gamma = self.v[gamma_key] / (1 - self.beta2 ** t)

            layer.gamma -= self.learning_rate * m_corrected_gamma / (np.sqrt(v_corrected_gamma) + self.eps)

        if hasattr(layer, 'grad_beta') and layer.grad_beta is not None:
            grad_beta = layer.grad_beta
            
            beta_key = f"{layer_id}_beta"
            if beta_key not in self.m:
                self.m[beta_key] = np.zeros_like(layer.beta)
                self.v[beta_key] = np.zeros_like(layer.beta)

            self.m[beta_key] = self.beta1 * self.m[beta_key] + (1 - self.beta1) * grad_beta
            self.v[beta_key] = self.beta2 * self.v[beta_key] + (1 - self.beta2) * (grad_beta ** 2)

            m_corrected_beta = self.m[beta_key] / (1 - self.beta1 ** t)
            v_corrected_beta = self.v[beta_key] / (1 - self.beta2 ** t)

            layer.beta -= self.learning_rate * m_corrected_beta / (np.sqrt(v_corrected_beta) + self.eps)


# ========== ЗАГРУЗЧИК ДАННЫХ ==========

class DataLoaderFixed:
    def __init__(self, X, y, batch_size=32, shuffle=False, drop_last=False):
        """
        Инициализация загрузчика данных

        Args:
            X: входные данные (features)
            y: метки классов (targets)
            batch_size: размер батча
            shuffle: перемешивать ли данные
            drop_last: отбрасывать ли последний неполный батч
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(X)
        
        self._reset()

    def _reset(self):
        """Сброс итератора"""
        self.indices = list(range(self.n_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        """Инициализация итератора"""
        self._reset()
        return self

    def __next__(self):
        """Получение следующего батча"""
        if self.current_idx >= self.n_samples:
            raise StopIteration()

        # Определяем размер текущего батча
        remaining = self.n_samples - self.current_idx
        current_batch_size = min(self.batch_size, remaining)
        
        # Если последний батч неполный и drop_last=True, пропускаем его
        if current_batch_size < self.batch_size and self.drop_last:
            raise StopIteration()

        # Получаем индексы для текущего батча
        batch_indices = self.indices[self.current_idx:self.current_idx + current_batch_size]
        
        # Извлекаем данные
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_idx += current_batch_size
        
        return batch_X.astype(np.float32), batch_y.astype(np.int32)

    def __len__(self):
        """Количество батчей в датасете"""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return int(np.ceil(self.n_samples / self.batch_size))


# ========== НЕЙРОННАЯ СЕТЬ ==========

class NeuralNetworkFixed:
    def __init__(self, batch_size=128, epochs=10, learning_rate=0.001):
        # Создание архитектуры нейронной сети
        self.model = Sequential(
            Linear(784, 512),
            BatchNorm(512),
            ReLU(),
            Dropout(0.4),
            
            Linear(512, 256),
            BatchNorm(256),
            ReLU(),
            Dropout(0.2),
            
            Linear(256, 128),
            BatchNorm(128),
            ReLU(),
            Dropout(0.2),
            
            Linear(128, 10)
        )
        
        # Инициализация оптимизатора с корректными параметрами
        self.optimizer = AdamFixed(learning_rate=learning_rate)
        
        # Инициализация функции потерь как экземпляр класса
        self.loss_func = CrossEntropyLoss()
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model.forward(x)

    def backward(self, grad_output):
        return self.model.backward(grad_output)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def get_trainable_layers(self):
        """Получение всех слоев с обучаемыми параметрами"""
        trainable_layers = []
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'grad_weight') or hasattr(layer, 'grad_gamma'):
                trainable_layers.append((i, layer))
        return trainable_layers

    def compute_loss(self, predictions, targets):
        """Вычисление функции потерь"""
        return self.loss_func.forward(predictions, targets)

    def compute_grad_loss(self):
        """Вычисление градиента функции потерь"""
        return self.loss_func.backward()

    def train_model(self, X, y):
        """Обучение модели"""
        # Создаем DataLoader для обучающих данных
        dataloader = DataLoaderFixed(X, y, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Массив для хранения значений loss
        losses = []
        
        self.train_mode()
        
        for epoch in range(self.epochs):
            epoch_losses = []
            print(f"Epoch {epoch + 1}/{self.epochs}")
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                # Очищаем градиенты
                self.zero_grad()
                
                # Прямое распространение
                predictions = self.forward(x_batch)
                
                # Вычисляем потери
                loss = self.compute_loss(predictions, y_batch)
                epoch_losses.append(loss)
                
                # Обратное распространение
                grad_loss = self.compute_grad_loss()
                self.backward(grad_loss)
                
                # Обновляем веса
                self.update_weights()
                
                if batch_idx % 5 == 0:  # Печатаем каждые 5 батчей
                    print(f"  Batch {batch_idx}, Loss: {loss:.4f}")
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"  Average Loss: {avg_loss:.4f}")
            print()
        
        return losses

    def zero_grad(self):
        """Обнуление градиентов всех слоев"""
        for layer in self.model.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

    def update_weights(self):
        """Обновление весов всех слоев"""
        trainable_layers = self.get_trainable_layers()
        for layer_id, layer in trainable_layers:
            self.optimizer.update(layer, layer_id)

    def predict(self, X):
        """Предсказание для данных"""
        self.eval_mode()
        
        # Создаем DataLoader для предсказаний
        dataloader = DataLoaderFixed(X, np.zeros(len(X)), batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        predictions = []
        for x_batch, _ in dataloader:
            result = self.forward(x_batch)
            batch_predictions = np.argmax(result, axis=1)
            predictions.extend(batch_predictions)
        
        return np.array(predictions)

    def evaluate_accuracy(self, X, y):
        """Вычисление точности на данных"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


# ========== СОЗДАНИЕ ДАННЫХ И ОБУЧЕНИЕ ==========

def create_synthetic_dataset(num_samples=1000, input_dim=784, num_classes=10, seed=42):
    """Создание синтетического датасета для тестирования нейронной сети"""
    np.random.seed(seed)
    
    # Создаем случайные входные данные
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    
    # Создаем синтетические метки с некоторой структурой
    # Добавляем небольшую корреляцию между входами и выходами
    weights = np.random.randn(input_dim, num_classes) * 0.01
    logits = X @ weights
    probabilities = softmax(logits)
    
    # Генерируем метки на основе вероятностей
    y = np.array([np.random.choice(num_classes, p=prob) for prob in probabilities])
    
    return X, y


def plot_training_curve(losses, title="Training Loss"):
    """Построение графика loss"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Добавляем аннотации для начального и конечного значения loss
        plt.annotate(f'Start: {losses[0]:.4f}', 
                    xy=(1, losses[0]), xytext=(len(losses)*0.2, losses[0]*1.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.annotate(f'End: {losses[-1]:.4f}', 
                    xy=(len(losses), losses[-1]), xytext=(len(losses)*0.8, losses[-1]*1.1),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
        
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        print("📊 График сохранен как 'training_loss.png'")
        plt.show()
    except Exception as e:
        print(f"⚠️ Не удалось построить график: {e}")
    
    # Выводим статистику
    print(f"Начальный loss: {losses[0]:.4f}")
    print(f"Конечный loss: {losses[-1]:.4f}")
    print(f"Снижение loss: {losses[0] - losses[-1]:.4f} ({((losses[0] - losses[-1])/losses[0]*100):.1f}%)")
    
    return losses[0] > losses[-1]  # Возвращаем True если loss падает


def train_and_evaluate_network():
    """Функция для создания, обучения и оценки нейронной сети"""
    print("🚀 Создание синтетического датасета...")
    
    # Создаем синтетические данные
    X, y = create_synthetic_dataset(num_samples=1000, input_dim=784, num_classes=10, seed=42)
    
    print(f"Размер датасета: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    print(f"Распределение классов: {np.bincount(y)}")
    
    # Разделяем данные на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Обучающая выборка: {X_train.shape}")
    print(f"Тестовая выборка: {X_test.shape}")
    
    print("\n🏗️ Создание нейронной сети...")
    
    # Создаем нейронную сеть
    network = NeuralNetworkFixed(
        batch_size=64,
        epochs=15,
        learning_rate=0.001
    )
    
    print("Архитектура сети:")
    for i, layer in enumerate(network.model.layers):
        print(f"  {i}: {layer.__class__.__name__}")
    
    print("\n🎯 Начинаем обучение...")
    
    # Обучаем сеть
    losses = network.train_model(X_train, y_train)
    
    print("✅ Обучение завершено!")
    
    # Строим график loss
    print("\n📊 Построение графика loss...")
    is_decreasing = plot_training_curve(losses, "Training Loss - Synthetic Dataset")
    
    # Оценка на тестовых данных
    print("\n🔍 Оценка на тестовых данных...")
    test_accuracy = network.evaluate_accuracy(X_test, y_test)
    train_accuracy = network.evaluate_accuracy(X_train, y_train)
    
    print(f"Точность на обучающих данных: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Точность на тестовых данных: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Проверяем критерий успешности
    print("\n🎯 Проверка критериев успешности:")
    print(f"✓ Loss падает: {'Да' if is_decreasing else 'Нет'}")
    print(f"✓ Обучение за {len(losses)} эпох: {'Да' if len(losses) <= 20 else 'Нет'}")
    
    success = is_decreasing and len(losses) <= 20
    print(f"\n{'🎉 УСПЕХ! Нейронная сеть успешно обучается!' if success else '❌ Требуется доработка.'}")
    
    return network, losses, train_accuracy, test_accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ НЕЙРОННОЙ СЕТИ")
    print("=" * 60)
    
    # Запускаем обучение
    network, losses, train_acc, test_acc = train_and_evaluate_network()