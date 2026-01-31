# DataForge Hackathon – AI Context Audit & Behavioral Anomaly Detection

This repository contains solutions for the **DataForge Pre-Evaluation Assignment** conducted by  
E-Summit’26, IIT Roorkee.

We implemented two independent but complementary systems:
 Problem Statement 1 – Self-Auditing Context Engine  
 Problem Statement 2 – Behavioral Anomaly Detection Engine  

Both systems focus on:
- Explainability
- Transparency
- No hard-coded rules
- Data-driven decision making
- CSV audit outputs

---

## 📁 Project Structure

DataForge-Hackathon-Solutions/
│
├── PS1_Context_Audit/
│   ├── context_scoring.py
│   ├── generate_csv.py
│   ├── sample_docs.json
│   └── output/
│       └── PS1_Context_Audit.csv
│
├── PS2_Behavioral_Anomaly/
│   ├── generate_db_activity_logs.py
│   ├── behavioral_anomaly_engine.py
│   └── output/
│       └── db_activity_logs.csv
│       └── anomaly_report.csv
│
├── assets/
│   ├── PS1_flowchart.jpg
│   └── PS2_flowchart.jpg
│
├── requirements.txt
└── README.md


#  Problem Statement 1 – Self-Auditing Context Engine

## 🎯 Objective
Given a user query and multiple document summaries, automatically determine which documents are relevant and generate a transparent audit report.

## 🧠 Approach
1. Load query and documents
2. Text preprocessing
3. TF-IDF vectorization
4. Cosine similarity scoring
5. Threshold filtering
6. Generate explainable CSV audit

## ⚙️ Techniques Used
- TF-IDF
- Cosine Similarity
- Threshold-based decision logic
- Explainable scoring

## 📤 Output
PS1_Context_Audit/output/PS1_Context_Audit.csv

Contains:
- query
- document_id
- similarity score
- threshold
- decision (Included/Excluded)
- reason


## 📊 Flowchart – Context Engine
This flowchart shows how the query and documents are processed using TF-IDF,
cosine similarity scoring, threshold filtering, and CSV audit generation.

<img src="assets/PS1_flowchart.jpg" width="900">

#  Problem Statement 2 – Behavioral Anomaly Detection Engine

## 🎯 Objective
Detect abnormal database activity by learning normal user behavior and identifying statistical deviations.

## 🧠 Approach
1. Generate database activity logs
2. Group logs by user
3. Learn baseline behavior (average rows scanned)
4. Compute Z-score deviation
5. Calculate risk percentile
6. Flag anomalies
7. Generate anomaly report

## ⚙️ Techniques Used
- Statistical modeling
- Z-score anomaly detection
- Percentile risk scoring
- Explainable alerts
- No if-else rule engine

## 📤 Output
PS2_Behavioral_Anomaly_Detection/output/PS2_Anomaly_Report.csv

Contains:
- baseline average
- z-score
- risk percentile
- risk level
- flagged (True/False)
- reason

## 📊 Flowchart – Behavioral Anomaly Detection
This flowchart shows how database logs are analyzed per user, baseline behavior
is learned, z-score is calculated, risk scoring is performed, and suspicious
activities are flagged.

<img src="assets/PS2_flowchart.jpg" width="900">

# ▶️ How to Run

## Install dependencies
pip install -r requirements.txt

## Run Problem Statement 1
cd PS1_Context_Audit python generate_csv.py

## Run Problem Statement 2
cd PS2_Behavioral_Anomaly_Detection python generate_db_activity_logs.py python behavioral_anomaly_engine.py


# 💡 Key Highlights

•Fully explainable decisions  
•No hardcoded rules  
• Data-driven approach  
• Modular design  
•Clean folder structure  
• Reproducible results  
• Industry-style audit logs  


# 🏁 Submission

This repository is submitted as part of the **DataForge Hackathon pre-evaluation (10% weightage)**.

Both solutions are runnable locally and generate their respective CSV outputs.

---

# 👨‍💻 Team
Team Name: Neural Forge 
Members: 4

