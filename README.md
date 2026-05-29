# 🧠 MNIST Digit Classifier — TensorFlow/Keras

A fully connected neural network for handwritten digit classification (0–9) using the MNIST dataset, built with TensorFlow/Keras and scikit-learn.

---

## 📌 About

This project implements a dense neural network to classify 28×28 grayscale images of handwritten digits. The model is trained using the Adam optimizer with sparse categorical crossentropy loss.

---

## 🗂️ Project Structure

```
.
├── dataset/
│   ├── train-images.idx3-ubyte   # Training images (IDX format)
│   └── train-labels.idx1-ubyte   # Training labels (IDX format)
├── main.py                        # Main script
└── README.md
```

---

## 🧱 Model Architecture

```
Input (28×28) → Flatten → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(10, Softmax)
```

| Layer   | Units | Activation |
|---------|-------|------------|
| Flatten | 784   | —          |
| Dense   | 64    | ReLU       |
| Dense   | 32    | ReLU       |
| Dense   | 16    | ReLU       |
| Dense   | 10    | Softmax    |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy

---

## ⚙️ Requirements

Python 3.8+ and the following libraries:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Download the MNIST dataset** in IDX format and place the files inside the `dataset/` folder:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`

   > Available at: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

3. **Run the script:**

```bash
python main.py
```

---

## 📊 Data Pipeline

- Reads binary IDX files with a custom `read_idx` parser
- Normalizes pixel values to the range `[0, 1]`
- Splits into train/test sets with an 80/20 ratio (`random_state=123`)

---

## 🏋️ Training Configuration

| Parameter    | Value          |
|--------------|----------------|
| Batch size   | 32             |
| Epochs       | 5              |
| Validation   | 20% of data    |
| Shuffle      | ✅             |

---

## 📈 Expected Results

With 5 epochs on MNIST, the model typically reaches **~97% validation accuracy**.

---

## 🛠️ Development

This project was developed using [Visual Studio Code](https://code.visualstudio.com/). The following extensions are recommended for a better experience:

| Extension | Purpose |
|-----------|---------|
| [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Linting, debugging, and IntelliSense |
| [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) | Fast type checking and autocompletion |
| [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) | Run and explore notebooks interactively |

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
