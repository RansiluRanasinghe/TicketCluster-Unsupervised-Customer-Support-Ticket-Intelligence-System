# 🎯 TicketCluster — Unsupervised Customer Support Ticket Intelligence System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-4285F4?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

TicketCluster is a **production-aligned unsupervised NLP system** that discovers hidden patterns in customer support tickets using **K-Means clustering**.

The project focuses on **modeling, evaluation, and system design**, demonstrating how organizations can analyze large volumes of unlabeled support tickets to uncover emerging issues and operational insights — **without relying on predefined categories**.

While the system was designed with API and cloud deployment readiness in mind, the project intentionally concludes at the **completed model + local inference stage**, reflecting a realistic ML development lifecycle.

> 🚫 Not a toy clustering demo  
> ✅ A **realistic unsupervised ML system** built with production constraints in mind

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
4. Designing a clean inference layer suitable for API-based systems

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
> This preserves true unsupervised learning behavior and mirrors real analytics pipelines.

---

## 🏗️ System Architecture (Conceptual)

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
Local Inference Layer
(API-ready design)
```

> **Note:** The system architecture is **cloud-ready by design**, though not deployed publicly.

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
- **Davies–Bouldin Index**
- **Inertia trends** (Elbow Method)
- **Cluster purity comparison** (labels used only for validation)

> **Note:** Moderate scores are expected — real support tickets are noisy, overlapping, and ambiguous.

---

## 🚀 API Design (Implemented Locally)

A **FastAPI-based inference layer** was implemented locally to demonstrate how clustering intelligence could be exposed in a **production-safe, stateless** manner.

### Core Capabilities

1. Accept raw customer ticket text
2. Apply the same preprocessing & vectorization pipeline
3. Assign tickets to discovered clusters
4. Return cluster metadata and similarity-based confidence signals

> **Development Status:** The API was developed and tested locally but intentionally **not deployed publicly**.

### Confidence Interpretation

**Confidence values** represent cosine similarity to the cluster centroid, **not classification certainty** — an important distinction in unsupervised systems.

---

## ☁️ Deployment Status

The system was designed for cloud deployment, but **public deployment was not completed**.

### Current Status

- ✅ **Cloud-ready architecture**
- ✅ **Stateless inference design**
- ✅ **Version-safe model loading**
- ❌ **No active AWS / Azure / Gitpod deployment**

This reflects a **realistic project boundary**, where development concludes after validating the full ML and system design pipeline.

---

## 🛠️ Tech Stack

### Machine Learning
- **Language**: Python
- **Framework**: scikit-learn
- **Vectorization**: TF-IDF
- **Algorithm**: K-Means Clustering

### Backend & System Design
- **API Framework**: FastAPI
- **Server**: Uvicorn (local)
- **Data Processing**: NumPy, Pandas

### Architecture
- **API-first design**
- **Stateless inference logic**
- **Cloud-compatible structure**

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.8+
- pip or conda

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

**Visit:** [http://localhost:8000/docs](http://localhost:8000/docs)

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

## 🧪 Example Usage (Local API)

### cURL
```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot login",
    "body": "Password reset link not working"
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/cluster",
    json={
        "subject": "Payment failed",
        "body": "Transaction declined but money was deducted"
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

## 🔮 Future Enhancements

- [ ] **Sentence-transformer embeddings** for deeper semantics
- [ ] **Dynamic cluster updates** for streaming data
- [ ] **Drift detection** for emerging issues
- [ ] **Persistent analytics storage**
- [ ] **Multilingual expansion**
- [ ] **Visualization dashboard** for cluster exploration
- [ ] **Public cloud deployment** (AWS / Azure / GCP)

---

## 📌 Why This Project Stands Out

This project demonstrates:

✓ **Unsupervised NLP modeling**  
✓ **Real-world support analytics**  
✓ **ML-to-API system design**  
✓ **Cloud-aware architecture thinking**  
✓ **Business-driven ML decisions**

### It Goes Beyond:

- ❌ Toy clustering demos
- ❌ Accuracy-obsessed notebooks
- ❌ Label-dependent ML systems

### Skills Demonstrated
- Unsupervised machine learning
- NLP and text processing
- RESTful API development with FastAPI
- Production-ready system architecture
- Business-focused ML evaluation

---

## 🎯 Use Cases

This system can be adapted for:
- **SaaS** — Support ticket intelligence
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

Always open to discussions on:
- Unsupervised learning in industry
- ML system design
- Support analytics
- Deployment best practices

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

**Built with production ML principles in mind.**

</div>
