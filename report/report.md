# **Sonar Rock vs Mine Classifier – Project Report**

## **1. Mission Brief**

The naval defense unit has tasked me with developing a system to protect our submarines from hidden underwater threats. Enemy forces are suspected of planting mines along the ocean bed to destroy passing submarines. These mines often resemble harmless rocks on SONAR scans, making it extremely difficult for human operators to tell the difference in time.

My mission: **Build a machine learning model that can analyze SONAR data and accurately determine whether a detected object is a mine or a rock.** This will give submarine crews a reliable, real-time decision-making tool to navigate safely, avoid dangerous zones, and protect both lives and military assets.

---

## **2. Objectives**

- Automate SONAR data interpretation.
    
- Reduce human error in mine detection.
    
- Enable real-time predictions onboard submarines.
    

---

## **3. Dataset Overview**

- **Source:** UCI Machine Learning Repository – _Sonar, Mines vs. Rocks Dataset_.
    
- **Features:** 60 continuous numerical attributes (sonar signal strength at various frequencies).
    
- **Labels:**
    
    - `"R"` → Rock
        
    - `"M"` → Mine
        
- **Total Samples:** 208
    
- **Class Balance:** Slightly imbalanced towards rocks.
    

---

## **4. Workflow**

### **4.1 Training Pipeline**

1. **SONAR Dataset**
    
2. **Train-Test Split** (90% train / 10% test, stratified)
    
3. **Logistic Regression Model Training**
    
4. **Model Evaluation** (Training & Test Accuracy)
    
5. **Model Saving** (`sonar_model.pkl`)
    

### **4.2 Prediction Pipeline**

1. **New SONAR Data Input** (60 features)
    
2. **Load Trained Model**
    
3. **Generate Prediction** (Rock or Mine)
    

![Workflow Diagram](images/workflow.png)
---

## **5. Implementation Details**

### **5.1 Dependencies**

From `requirements.txt`:

```
ipykernel==6.29.5
matplotlib==3.10.0
numpy==2.2.5
pandas==2.3.1
scikit-learn==1.7.1
joblib==1.5.1
```

### **5.2 Training**

- Model: Logistic Regression.
    
- Accuracy:
    
    - **Training:** ~83.42%.
        
    - **Test:** ~76.19%.
        

### **5.3 Prediction Example**

Sample Input Data (first 5 features shown):

```
0.0270, 0.0163, 0.0346, 0.0216, 0.0637, ...
```

Prediction Output:

```
Prediction: The object is a Rock :)
```

---

## **6. Future Improvements**

- **Integrate with real-time SONAR feeds** so predictions are instant.
    
- **Experiment with advanced ML models** such as Random Forests, Gradient Boosting, or Neural Networks.
    
- **Implement noise reduction & feature scaling** to handle variable oceanic conditions.
    
- **Develop a dashboard interface** for crews to see predictions and confidence scores.
    
- **Add anomaly detection** for previously unseen patterns.
    
- **Optimize for low-latency onboard processing** in submarines.
    

---

## **7. Strategic Impact**

- **Rapid Threat Detection:** Reduces risk from underwater mines.
    
- **Operational Safety:** Protects both personnel and military assets.
    
- **Mission Success Rate:** Supports strategic submarine navigation.
    

---

## **8. Conclusion**

This project demonstrates the effectiveness of machine learning in high-stakes defense applications. By leveraging SONAR data and a Logistic Regression classifier, we have created a reliable tool for detecting mines and rocks — enhancing naval safety and mission efficiency.
