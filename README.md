Этот код представляет собой простую реализацию нейронной сети с одним скрытым слоем для распознавания рукописных цифр из набора данных MNIST. Давайте разберем, что делает каждая часть кода:

Здесь импортируются библиотеки numpy для работы с массивами, matplotlib для визуализации данных и пользовательский модуль utils, который, предположительно, содержит функцию load_dataset для загрузки данных MNIST.

Загрузка данных:


images, labels = utils.load_dataset()

Загружаются изображения и соответствующие им метки из набора данных MNIST с помощью функции load_dataset из модуля utils.

Инициализация весов и смещений:


weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

Инициализируются случайные веса для связей между входным и скрытым слоем и между скрытым и выходным слоями, а также нулевые смещения для скрытого и выходного слоев.

Обучение нейронной сети:


epochs = 3
learning_rate = 0.01

for epoch in range(epochs):
    for image, label in zip(images, labels):
        # Прямое распространение
        # Обратное распространение
    # Вывод информации о потерях и точности после каждой эпохи

Происходит обучение нейронной сети в течение заданного количества эпох. В каждой эпохе происходит проход по всем изображениям из набора данных и обновление весов с помощью обратного распространения ошибки.

Визуализация результата:


import random

test_image = random.choice(images)

image = np.reshape(test_image, (-1, 1))

hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))

output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the number is: {output.argmax()}")
plt.show()

Выбирается случайное изображение из набора данных, после чего прогоняется через нейронную сеть для получения прогноза. Изображение выводится на экран с указанием предполагаемой цифры, предсказанной нейронной сетью.
