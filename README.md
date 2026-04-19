# Towards Robust CAPTCHA Design : A Multi-Agent Curriculum Learning Approach with Calibration and Fairness Analysis

---

## 📌 Overview

This project presents an adaptive CAPTCHA framework designed to address the limitations of traditional CAPTCHA systems in the presence of advanced AI-based attacks such as OCR and deep learning models.

Conventional CAPTCHA systems either become:

* Easily bypassable by intelligent bots
* Or excessively complex for human users

This work proposes a closed-loop, learning-based system that dynamically balances security and usability.

---

## 🔐 Problem Statement

Modern automated systems can effectively break static CAPTCHA mechanisms using machine learning techniques. Increasing CAPTCHA difficulty improves security but negatively impacts user experience, leading to a fundamental trade-off between usability and robustness.

---

## ⚙️ Proposed Solution

We design an adaptive CAPTCHA framework based on:

* Multi-agent adversarial modeling (weak to strong attackers)
* Curriculum learning for progressive difficulty adaptation
* Human behavior proxy (accuracy, response time, cognitive load)
* Entropy-based calibration for reliability

---

## 🧠 Key Innovation

### Adaptive Robustness Index (ARI)

A unified evaluation metric defined as:

ARI = α · (Human Score) + β · (1 − Bot Success) − γ · (Difficulty)

This metric jointly captures:

* Human usability
* Resistance to automated attacks
* Challenge complexity

---

## 🏗️ System Pipeline

Challenge Generator → Attacker Model → Human Model → ARI Computation → Curriculum Update

---

## 📁 Project Structure

```
CAP_PROJ/
│── core/
│   ├── attacker.py
│   ├── curriculum.py
│   ├── generator.py
│   ├── human_model.py
│   ├── metrics.py
│
│── engine/
│   ├── simulator.py
│   ├── logger.py
│
│── analysis/
│   ├── evaluator.py
│   ├── metrics.py
│   ├── validation_metrics.py
│
│── models/
│   ├── trainer.py
│
│── results/
│── plots/
│── real_captcha_samples/
│
│── main.py
│── config.py
│── captcha_image_generator.py
│── real_world_eval.py
│── visualization.py
│── human_ui.py
│── verification_log.csv
│── README.md
```

---

## ⚙️ Methodology

### Challenge Generation

Controlled parameters:

* warp
* clutter
* variation
* entropy

### Multi-Agent Attacker

Simulates attackers of varying strengths. Bot success is modeled as:

```
bot_success = exp(-12 × difficulty × strength)
```

### Human Behavior Proxy

Models realistic human interaction using:

* accuracy
* response time
* cognitive load

### Curriculum Learning

Adaptive update rule:

```
delta = (target_ari - ari) + entropy_weight × entropy
```

---

## 📊 Results

### Performance Metrics

* Mean ARI: 0.7775
* Bot Success Rate: 0.0646
* Human Accuracy: ~0.90
* Average Response Time: 3.48 sec
* Stability (Std): 0.0278

### Improvement Over Baseline

* Baseline bot success: 0.30
* Proposed system: 0.0646
* Reduction: 78.46%

### Robustness & Validation

* Worst-case ARI: 0.7689
* Calibration Error: 0.0018
* Effect Size (Cohen’s d): 6.14
* 95% Confidence Interval: [0.7767, 0.7783]

### Fairness Analysis

| Difficulty | Human Score |
| ---------- | ----------- |
| Easy       | 0.8631      |
| Medium     | 0.7987      |
| Hard       | 0.7654      |

---

## 🌍 Real-World Evaluation

* OCR evaluated using Tesseract
* Average OCR Accuracy: 0.0645
* Maximum OCR Accuracy: 0.5455
* Human Accuracy: ~90%

The system demonstrates strong resistance to automated attacks while maintaining usability.

---

## ▶️ How to Run

### 1. Clone Repository

```
git clone https://github.com/sangwanmukul/capstone_project.git
cd capstone_project
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Project

```
python main.py
```

---

## 📊 Outputs

* `results/experiment.csv` — simulation logs
* `results/failure_cases.csv` — failure analysis
* Generated plots:

  * ARI curve
  * Calibration curve
  * Stability analysis
  * Fairness evaluation
  * Performance trade-off

---

## 🛠️ Tech Stack

* Python
* NumPy, Pandas
* LightGBM
* OpenCV
* Matplotlib
* Tesseract OCR

---

## 🚀 Contributions

* Adaptive CAPTCHA system using curriculum learning
* Multi-agent adversarial simulation
* Unified ARI evaluation metric
* Fairness, calibration, and robustness analysis
* Real-world OCR validation

---

## 👨‍💻 Author

Mukul Sangwan
Final Year Computer Science Engineering Student

---

## 📜 License

This project is intended for academic and research purposes.
