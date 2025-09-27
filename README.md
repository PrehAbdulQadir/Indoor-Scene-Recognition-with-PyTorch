# Indoor-Scene-Recognition-with-PyTorch
Indoor Scene Recognition with PyTorch – Classifying 67 different indoor environments (airport, library, restaurant, etc.) using deep learning and transfer learning.

# PyScene67: Indoor Scene Recognition with PyTorch

This project implements an **Indoor Scene Recognition** model using **PyTorch**, trained on the CVPR 2019 Indoor Scenes dataset. The dataset includes 67 different indoor scene categories, ranging from **meeting rooms** and **libraries** to **restaurants** and **airports**.

---

## 📂 Dataset

* Source: [Kaggle Indoor Scenes CVPR 2019](https://www.kaggle.com/)
* Path: `/kaggle/input/indoor-scenes-cvpr-2019`

**Structure:**

```
train: ../train/images
val:   ../valid/images
test:  ../test/images
```

**Classes (67 total):**

```
['meeting_room', 'grocerystore', 'trainstation', 'mall', 'bar', 'auditorium',
 'laboratorywet', 'florist', 'closet', 'livingroom', 'pantry', 'airport_inside',
 'prisoncell', 'locker_room', 'elevator', 'dentaloffice', 'laundromat',
 'fastfood_restaurant', 'casino', 'dining_room', 'kindergarden', 'concert_hall',
 'waitingroom', 'bathroom', 'corridor', 'bedroom', 'hairsalon', 'kitchen',
 'tv_studio', 'artstudio', 'library', 'inside_bus', 'restaurant_kitchen',
 'inside_subway', 'buffet', 'bookstore', 'museum', 'lobby', 'gameroom',
 'shoeshop', 'garage', 'poolinside', 'clothingstore', 'deli', 'subway',
 'jewelleryshop', 'stairscase', 'toystore', 'classroom', 'restaurant',
 'nursery', 'bakery', 'bowling', 'office', 'operating_room', 'warehouse',
 'studiomusic', 'church_inside', 'computerroom', 'cloister', 'greenhouse',
 'winecellar', 'gym', 'videostore', 'hospitalroom', 'children_room', 'movietheater']
```

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/pyscene67.git
cd pyscene67
pip install -r requirements.txt
```

---

## 🚀 Training

To train the model:

```bash
python train.py --epochs 30 --batch-size 64 --lr 0.001
```

---

## 📊 Evaluation

Run evaluation on the validation or test set:

```bash
python evaluate.py --weights best_model.pth --data ../valid/images
```

---

## 🧠 Model

* Framework: **PyTorch**
* Backbone: (e.g., ResNet50, EfficientNet, or custom CNN)
* Loss: CrossEntropy
* Optimizer: Adam / SGD

---

## 📈 Results

<img width="240" height="160" alt="Screenshot 2025-09-27 150826" src="https://github.com/user-attachments/assets/f4722f60-d812-4084-bfa0-cfdef7684ae3" />
<img width="245" height="170" alt="Screenshot 2025-09-27 150903" src="https://github.com/user-attachments/assets/cd78675b-7fcd-4b22-ba84-774d7bc65f0d" />
<img width="209" height="187" alt="Screenshot 2025-09-27 150923" src="https://github.com/user-attachments/assets/c7fbbf0b-a165-4336-a2b5-47d8741ede6b" />
<img width="212" height="185" alt="Screenshot 2025-09-27 150940" src="https://github.com/user-attachments/assets/81993284-d5a7-4c3f-9dbd-98f3d67d7c8b" />
<img width="176" height="184" alt="Screenshot 2025-09-27 150957" src="https://github.com/user-attachments/assets/9c767bf2-0cc1-4f78-a81b-563e87f6473a" />
<img width="179" height="184" alt="Screenshot 2025-09-27 151028" src="https://github.com/user-attachments/assets/5c68e2b6-c794-4e4b-b504-33cab510be95" />


---

## 🔮 Future Work

* Experiment with **Vision Transformers (ViT)** and **Swin-Transformer**.
* Apply **data augmentation** for improved generalization.
* Use **transfer learning** from ImageNet-pretrained models.
* Add **Grad-CAM** for model explainability.

---




