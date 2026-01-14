# 🎯 TicketCluster — Unsupervised Customer Support Ticket Intelligence System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

TicketCluster is a **production-aligned unsupervised NLP system** that discovers hidden patterns in customer support tickets using **K-Means clustering**, exposes insights via a **FastAPI service**, and is designed for **cloud-native deployment**.

This project mirrors how real organizations analyze large volumes of unlabeled support tickets to uncover emerging issues, reduce manual analysis, and support operational decision-making — **without relying on predefined categories**.

> 🚫 Not a toy clustering demo  
> ✅ A **realistic ML system** built with production constraints in mind

---

## 🧠 Business Problem

Modern organizations receive thousands of customer support tickets daily. In real environments, many tickets:

- ❌ Are **unlabeled**
- ❌ Come from **new products, features, or regions**
- ❌ Represent **emerging issues** not yet categorized

### Challenges in Real Systems

- Manual analysis **does not scale**
- Labeling is **expensive and slow**
- Predefined categories **lag behind real customer problems**

### The Need

Organizations require systems that can:
- ✅ Detect **emerging issue patterns**
- ✅ Group **similar tickets automatically**
- ✅ Support **human decision-making** with data

---

## 💡 Solution Overview

**TicketCluster** addresses this problem by:

1. Converting raw ticket text into numerical representations
2. Applying unsupervised K-Means clustering
3. Discovering natural semantic groupings of customer issues
4. Serving clustering intelligence through a clean FastAPI service

### Business Value

This enables:
- 🔍 **Early issue discovery**
- 📊 **Support workload and trend analysis**
- 💡 **Operational insight** without labeled data

---

## 📊 Dataset

| Property | Description |
|----------|-------------|
| **Source** | Kaggle — Multilingual Customer Support Tickets |
| **Type** | Synthetic but industry-realistic |
| **Languages** | English, German (extensible) |
| **Fields Used** | Subject, Body |
| **Labels** | Not used for training |

### Important Note

> **Existing labels** (e.g., queues) are used **only for evaluation and validation**, never for training.  
> This preserves true unsupervised learning behavior, matching real-world analytics pipelines.

---

## 🏗️ System Architecture

```
Customer Ticket Text
(Subject + Body)
        ↓
Minimal Text Preprocessing
(Production-safe)
        ↓
TF-IDF Vectorization
        ↓
K-Means Clustering
(Unsupervised)
        ↓
FastAPI Inference Layer
        ↓
Cloud Deployment (AWS-ready)
```

---

## 🧠 Model Design

### Why K-Means?

- ✓ **Industry-standard** clustering baseline
- ✓ **Interpretable** cluster centroids
- ✓ **Scalable** for large ticket volumes
- ✓ **Easy to audit** and explain to stakeholders

### Feature Engineering

- **Combined text**: Subject + Body
- **Minimal preprocessing** (no aggressive cleaning)
- **TF-IDF with n-grams** for semantic signal

### Cluster Interpretation

Clusters are analyzed and assigned **human-readable business themes** using:
- Dominant centroid keywords
- Sample ticket inspection
- Optional comparison with known queues

This mirrors how **real analytics teams** interpret unsupervised results.

---

## 📈 Evaluation Strategy

Because this is an **unsupervised system**:

- ❌ No accuracy score is optimized
- ✅ Evaluation focuses on:
  - **Cluster coherence**
  - **Semantic consistency**
  - **Business interpretability**

### Supporting Analysis

- **Silhouette Score**
- **Davies-Bouldin Index**
- **Inertia trends** (Elbow Method)
- **Cluster purity comparison** (labels used only for validation)

> **Note:** Moderate scores are expected — real support tickets are noisy, overlapping, and ambiguous.

---

## 🚀 API Design (FastAPI)

The FastAPI layer exposes clustering intelligence in a **production-safe, stateless** manner.

### Core Capabilities

1. Accept raw customer ticket text
2. Apply the same preprocessing & vectorization pipeline
3. Assign tickets to discovered clusters
4. Return cluster metadata and confidence signals

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
  "total_clusters": 6,
  "uptime_seconds": 1234.5
}
```

---

### Cluster a Ticket
```http
POST /cluster
```

**Request:**
```json
{
  "subject": "Cannot login to my account",
  "body": "I've reset my password multiple times but still get an authentication error",
  "language": "en",
  "priority": 2
}
```

**Response:**
```json
{
  "cluster_id": 5,
  "cluster_theme": "Data Sync & Integration Issues",
  "confidence": 0.06,
  "similar_keywords": ["account", "login", "documents"],
  "sample_size": 1000,
  "processing_time_ms": 65.0
}
```

> **Note:** Confidence represents cosine similarity to the cluster centroid, not classification certainty.

---

### Batch Clustering
```http
POST /cluster/batch
```

Processes multiple tickets efficiently for analytics workflows.

**Request:**
```json
{
  "tickets": [
    {
      "subject": "Payment failed",
      "body": "Transaction declined"
    },
    {
      "subject": "App crash",
      "body": "Crashes when uploading files"
    }
  ]
}
```

---

### Cluster Metadata
```http
GET /clusters
```

Returns discovered cluster themes, sizes, and top keywords.

**Response:**
```json
{
  "total_clusters": 6,
  "clusters": [
    {
      "cluster_id": 0,
      "theme": "Billing & Payments",
      "size": 1247,
      "top_keywords": ["payment", "charge", "invoice"]
    }
  ]
}
```

---

## 🧪 Example Usage

### cURL
```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Payment failed",
    "body": "Transaction was declined but money was deducted"
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/cluster",
    json={
        "subject": "Cannot login",
        "body": "Password reset link not working"
    }
)

print(response.json())
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/cluster', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    subject: 'App keeps crashing',
    body: 'Crashes every time I upload files'
  })
});

const result = await response.json();
console.log(result);
```

---

## ☁️ Deployment Strategy

The system is designed with **AWS deployment readiness** in mind:

- ✓ Model artifacts stored externally (**S3-ready**)
- ✓ **Stateless FastAPI** inference
- ✓ **Version-safe** model loading
- ✓ **Public endpoint** compatibility

### Deployment Architecture

```
S3 (Model Artifacts)
        ↓
EC2 / ECS (FastAPI)
        ↓
API Gateway
        ↓
CloudWatch
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
- **Cloud Provider**: AWS (deployment-ready)
- **Storage**: S3 (artifact storage)
- **Architecture**: Stateless API design

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- AWS account (optional)

### Run Locally

```bash
# Clone repository
git clone https://github.com/RansiluRanasinghe/TicketCluster-Unsupervised-ML.git
cd TicketCluster-Unsupervised-ML

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Discovered Clusters (Example)

| Cluster | Theme | Description |
|---------|-------|-------------|
| 0 | Billing & Payments | Charges, invoices, refunds |
| 1 | Technical Errors | Crashes, bugs, failures |
| 2 | Authentication | Login & password issues |
| 3 | Account Management | Profile & settings |
| 4 | Performance Issues | Slow response & timeouts |
| 5 | General Support | Mixed or emerging issues |

---

## 🔮 Future Enhancements

- [ ] **Sentence-transformer embeddings** for better semantic understanding
- [ ] **Dynamic cluster updates** as new tickets arrive
- [ ] **Drift detection** for emerging issues
- [ ] **Persistent analytics storage**
- [ ] **Multilingual expansion**
- [ ] **Visualization dashboard** for cluster exploration

---

## 📌 Why This Project Stands Out

This project demonstrates:

✓ **Unsupervised NLP modeling**  
✓ **Real-world support analytics**  
✓ **ML-to-API system design**  
✓ **Cloud deployment awareness**  
✓ **Business-driven ML thinking**

### It Goes Beyond:

- ❌ Toy clustering demos
- ❌ Accuracy-obsessed notebooks
- ❌ Label-dependent ML systems

### Skills Demonstrated
- Unsupervised machine learning
- NLP and text processing
- RESTful API development
- Cloud architecture (AWS)
- Production ML system design

---

## 🎯 Use Cases

This system can be adapted for:
- **SaaS** — Support ticket intelligence
- **E-commerce** — Inquiry analysis
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

Always open to discussions on:
- Unsupervised learning in industry
- ML system design
- Support analytics
- Deployment best practices

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

**Built with a production-first ML mindset.**

</div>
