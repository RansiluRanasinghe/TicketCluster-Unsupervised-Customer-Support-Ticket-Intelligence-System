# 🎯 TicketCluster — Unsupervised Customer Support Ticket Intelligence System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

TicketCluster is an **unsupervised NLP system** that automatically discovers hidden patterns in customer support tickets using **K-Means clustering**, exposes the intelligence via a **FastAPI service**, and is designed for **cloud deployment**.

This project simulates how real companies analyze large volumes of unlabeled support tickets to improve routing, operations, and decision-making — **without relying on pre-defined categories**.

> 🚫 Not a toy clustering demo  
> ✅ A **production-aligned ML system** for real operational use cases

---

## 🧠 Business Problem

Modern organizations receive thousands of customer support tickets daily, many of which:

- ❌ Are **unlabeled**
- ❌ Come from **new products or regions**
- ❌ Represent **emerging issues** not yet categorized

### Challenges in Real Systems

- Manual analysis **does not scale**
- Labeling is **expensive and slow**
- Predefined categories **lag behind real problems**

### The Need

Companies require systems that can:
- ✅ Detect **emerging issue patterns**
- ✅ Group **similar tickets automatically**
- ✅ Support **human decision-making** with data

---

## 💡 Solution Overview

**TicketCluster** addresses this problem by:

1. Converting raw ticket text into numerical representations
2. Applying unsupervised K-Means clustering
3. Discovering natural groupings of customer issues
4. Serving clustering results via a clean, production-ready API

### Business Value

This enables:
- 🔍 **Early issue discovery**
- 📊 **Support workload analysis**
- 💡 **Operational insight** without labeled data

---

## 📊 Dataset

| Property | Description |
|----------|-------------|
| **Source** | Kaggle — Multilingual Customer Support Tickets |
| **Type** | Synthetic but industry-realistic support tickets |
| **Languages** | English, German (extensible) |
| **Fields Used** | Subject, Body |
| **Labels** | Not used during training (unsupervised learning) |

### Important Note

> **Existing labels** (e.g., queues) are used **only for evaluation and validation**, never for model training.  
> This ensures the system reflects **true unsupervised learning behavior**.

---

## 🏗️ System Architecture

```
┌──────────────────────┐
│ Customer Ticket Text │
│ (Subject + Body)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│ Text Preprocessing       │
│ (Minimal, Production)    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Vectorization (TF-IDF)   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ K-Means Clustering       │
│ (Unsupervised Model)     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ FastAPI Inference Layer  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ AWS Deployment           │
│ (Public Access)          │
└──────────────────────────┘
```

---

## 🧠 Model Design

### Why K-Means?

- ✓ **Industry-standard** clustering baseline
- ✓ **Interpretable** cluster centroids
- ✓ **Scalable** for large ticket volumes
- ✓ **Easy to audit** and explain

### Feature Engineering

- **Combined text**: Subject + Body
- **Minimal preprocessing** (no aggressive cleaning)
- **TF-IDF vectorization** for semantic representation

### Cluster Interpretation

Clusters are analyzed and assigned **human-readable themes** based on:
- Dominant keywords
- Sample ticket inspection
- Business relevance

This mirrors how **real analytics teams** work.

---

## 📈 Evaluation Strategy

Since this is an **unsupervised system**:

- ❌ No accuracy score is optimized
- ✅ Evaluation focuses on:
  - **Cluster coherence**
  - **Semantic consistency**
  - **Business interpretability**

### Optional Analysis

- Comparing discovered clusters with existing ticket queues
- Identifying overlaps and hidden sub-themes
- Validating cluster themes with domain experts

---

## 🚀 API Design (FastAPI)

### Core Capabilities

1. Accept raw customer ticket text
2. Apply the same preprocessing pipeline
3. Assign the ticket to a discovered cluster
4. Return cluster metadata

### Example API Response

```json
{
  "cluster_id": 3,
  "cluster_theme": "Authentication & Login Issues",
  "similarity_score": 0.87,
  "timestamp": "2025-01-07T10:45:00Z"
}
```

### Why This Matters

This demonstrates:
- ✓ **ML-to-production** thinking
- ✓ **Model serving** practices
- ✓ **Clean API contract** design

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "total_clusters": 7
}
```

---

### Cluster Ticket
```http
POST /cluster
```

**Request Body:**
```json
{
  "subject": "Cannot reset password",
  "body": "I've tried multiple times but the reset link doesn't work"
}
```

**Response:**
```json
{
  "cluster_id": 3,
  "cluster_theme": "Authentication & Login Issues",
  "similarity_score": 0.87,
  "top_keywords": ["password", "reset", "login", "authentication"],
  "timestamp": "2025-01-07T10:45:00Z"
}
```

---

### Get Cluster Statistics
```http
GET /clusters/stats
```

**Response:**
```json
{
  "total_clusters": 7,
  "clusters": [
    {
      "cluster_id": 0,
      "theme": "Billing & Payment Issues",
      "size": 1247,
      "percentage": 18.5
    },
    {
      "cluster_id": 1,
      "theme": "Technical Support",
      "size": 1098,
      "percentage": 16.3
    }
  ]
}
```

---

## 🧪 Example Usage

### cURL
```bash
curl -X POST "http://localhost:8000/cluster" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Payment failed",
    "body": "Transaction was declined but money was deducted"
  }'
```

### Python
```python
import requests

url = "http://localhost:8000/cluster"
data = {
    "subject": "Cannot login",
    "body": "Password reset link not working"
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/cluster', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    subject: 'App keeps crashing',
    body: 'Every time I try to upload files the app crashes'
  })
});

const result = await response.json();
console.log(result);
```

---

## ☁️ Deployment Strategy

The project is designed with **AWS deployment readiness** in mind:

- ✓ Model artifacts stored in **S3**
- ✓ **Stateless inference** logic
- ✓ API designed for **scalability**
- ✓ **Public accessibility** for demos and testing

This mirrors internal ML services used in **enterprise environments**.

### Deployment Architecture

```
AWS S3 (Model Storage)
        │
        ▼
EC2 / ECS (FastAPI Service)
        │
        ▼
API Gateway (Public Endpoint)
        │
        ▼
CloudWatch (Monitoring)
```

---

## 🛠️ Tech Stack

### Machine Learning
- **Language**: Python
- **Framework**: scikit-learn
- **Vectorization**: TF-IDF
- **Algorithm**: K-Means Clustering

### Backend
- **API Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **Data Processing**: NumPy, Pandas

### Cloud & Infrastructure
- **Cloud Provider**: AWS
- **Storage**: S3 (artifact hosting)
- **Architecture**: Deployment-ready, stateless

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- AWS account (optional, for deployment)
- pip or conda

### 1️⃣ Clone Repository
```bash
git clone https://github.com/RansiluRanasinghe/TicketCluster-Unsupervised-ML.git
cd TicketCluster-Unsupervised-ML
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train Clustering Model
```bash
jupyter notebook
# Open and run clustering notebook
```

### 4️⃣ Start API Server
```bash
uvicorn main:app --reload
```

### 5️⃣ Test API
Navigate to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Discovered Clusters (Example)

| Cluster ID | Theme | Top Keywords | Size |
|------------|-------|--------------|------|
| 0 | Billing & Payments | payment, charge, invoice, refund | 18.5% |
| 1 | Technical Support | error, crash, bug, issue | 16.3% |
| 2 | Authentication | login, password, reset, access | 14.2% |
| 3 | Account Management | account, profile, settings, update | 12.8% |
| 4 | Product Features | feature, function, how-to, guide | 11.7% |
| 5 | Performance Issues | slow, loading, timeout, speed | 10.9% |
| 6 | General Inquiries | question, information, help, support | 15.6% |

---

## 🔮 Future Enhancements

- [ ] Add **sentence-transformer embeddings** for better semantic clustering
- [ ] Implement **dynamic cluster updates** as new tickets arrive
- [ ] Add **drift detection** for emerging issues
- [ ] Integrate **storage layer** for ticket analytics
- [ ] Extend to **multilingual clustering**
- [ ] Add **visualization dashboard** for cluster exploration

---

## 📌 Why This Project Stands Out

This project demonstrates:

✓ **Unsupervised NLP modeling**  
✓ **Real-world customer support analytics**  
✓ **Production-oriented API design**  
✓ **Cloud deployment awareness**  
✓ **Business-driven ML decisions**

### It Goes Beyond:

- ❌ Simple clustering demos
- ❌ Academic notebooks
- ❌ Label-dependent ML projects

### Skills Demonstrated
- Unsupervised machine learning
- NLP and text processing
- API development with FastAPI
- Cloud architecture (AWS)
- Production ML system design

---

## 🎯 Use Cases

This system can be adapted for:
- **SaaS companies** — Support ticket intelligence
- **E-commerce** — Customer inquiry analysis
- **Healthcare** — Patient feedback clustering
- **Finance** — Complaint pattern detection
- **Product teams** — Feature request grouping

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

**Interests:**  
Machine Learning • NLP • Backend Engineering • Production ML Systems

Always open to discussions around:
- Unsupervised learning in industry
- ML system design
- Support analytics
- Deployment best practices

---

<div align="center">

**⭐ If you find this project valuable, consider giving it a star!**

**Built with a production-first ML mindset.**

</div>
