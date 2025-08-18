# 🏋️‍♀️ AI Fitness Coach – Action Recognition  

## 📌 Project Overview  
This project is about **recognizing fitness exercises from video recordings** using skeleton data.  
The idea: a user records themselves performing an exercise, and the model detects **which movement was performed**.  
Future steps: after detecting the movement, the model will also evaluate whether the exercise was performed correctly.  

## ⚙️ How It Works  
1. **Data Extraction** – Using [MediaPipe](https://mediapipe.dev/), skeleton landmarks (joint coordinates) are extracted frame by frame from the video.  
2. **Preprocessing**  
   - Normalization of skeleton coordinates.  
   - Padding of sequences to unify video length.  
   - Masking to ignore padded (empty) frames.  
3. **Model Architecture**  
   - **Input:** sequence of frames with joint coordinates.  
   - **Masking Layer:** skips padded frames.  
   - **Two stacked LSTM layers:** capture temporal dynamics of the movement.  
   - **Dense + Softmax:** outputs probability distribution over exercise classes.  
4. **Training Pipeline** – Implemented with `tf.data` for efficient batch loading and memory management.  

## 🛠️ Challenges & Solutions  
- **Variable sequence length:** solved with padding + masking.  
- **Large dataset & limited hardware:** solved by using small batch size and `tf.data` streaming.  
- **Stability of training:** solved via normalization of features.  

## 🚀 Technologies  
- Python  
- TensorFlow / Keras  
- MediaPipe  
- NumPy  

## 📊 Results  
- Successfully detects the exercise performed based on skeleton data.  
- Provides a foundation for the next step: **evaluating the quality of the exercise execution**.  

## 🔮 Future Work  
- Add a second model to **evaluate correctness of execution**.  
- Expand dataset with more diverse exercises.  
- Optimize for real-time feedback inside a fitness application.  

## 👩‍💻 Author  
Developed by **Abigail Berkk**  
