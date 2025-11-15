American Sign Language (ASL) Detection)
🎯 Goal
Built a deep-learning model that recognizes ASL hand signs (A–Z + SPACE, DELETE, NOTHING) from images.
Used MobileNetV2 to keep it lightweight and fast enough to train on my CPU.
📂 Dataset
From Kaggle – ASL Alphabet Dataset
29 total classes (A–Z + 3 extra signs)
Around 3K images per class in the training folder
⚙️ Model Details
Base Model: MobileNetV2 (transfer learning)
Image Size: 64×64
Epochs: 5 (CPU-friendly)
Optimizer: Adam
📊 Results
Metric	Score
Training Accuracy	~91%
Validation Accuracy	~64%
👉 Shows a bit of overfitting — the model learns training data really well but struggles a bit on unseen data.
Totally expected since I trained only 5 epochs on a normal CPU.
With a stronger setup (GPU or more time), accuracy can easily go 85–95%.
🧠 Quick ML Talk
Underfitting: Model didn’t learn enough → both accuracies low.
Overfitting: Model learned too much from training → big accuracy gap.
Mine’s slightly overfitted — small system, less training time. Still performs well overall.
🚀 Future Plans
Train longer on GPU.
Add early stopping & data augmentation.
Fine-tune more layers of MobileNetV2.
💻 App
Made a simple Streamlit app (app.py) — just upload an ASL image and it predicts the alphabet.
Run it with:
streamlit run app.py
💬 Final Note
The project works solidly as a prototype.
It proves that even on limited hardware, you can build an ASL detection system that performs decently — and it’ll only get better with more power and training time.