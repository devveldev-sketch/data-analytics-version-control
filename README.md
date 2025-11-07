# FDA22 — Real-Time Competitor Analysis System

This repository contains the **Forensic Data Analytics (FDA22)** project developed to demonstrate how **version control**, **AI**, and **data analytics** can be integrated to build a real-time **Competitor Analysis System**.

The project uses **Streamlit**, **Cohere API**, **NewsAPI**, and **Twitter API** to analyze companies dynamically, identify real competitors, and generate live **SWOT (Strengths, Weaknesses, Opportunities, Threats)** insights based on online data.

---

## 📘 Project Overview

Conventional competitor analysis tools rely on static or manually curated data.  
This project transforms that process into a **dynamic, AI-driven system** that collects real-time data from news, social media, and APIs to perform intelligent market assessments.

Using version control (Git and GitHub), the entire analytics workflow is tracked — from code updates to dataset versioning — ensuring **reproducibility** and **collaboration**.

---

## 🎯 Objectives

- To demonstrate the role of **version control** in managing a data analytics workflow.  
- To develop an **AI-enabled real-time competitor analysis system**.  
- To automatically identify a company’s **industry** and **real competitors**.  
- To analyze live market data to produce dynamic **SWOT analysis** reports.  
- To maintain the project through structured commits and GitHub synchronization.

---

## ⚙️ System Architecture

User Input → Industry Detection → Competitor Identification → Data Collection (API)
→ Text Embedding (Cohere) → Similarity Analysis → SWOT Generation → Streamlit Dashboard

---

**Data Sources:**
- **Cohere API:** Text understanding & embeddings  
- **NewsAPI:** Fetches latest company-related news articles  
- **Twitter API:** Extracts social media insights  
- **NumPy + httpx:** Handles numerical processing & async requests

---

## 🧩 Features

### 🔹 Real-Time Competitor Discovery
Automatically identifies relevant companies in the same sector using intelligent keyword and context analysis.

### 🔹 Industry Classification
Supports multiple industries:
- E-commerce Platforms (Shopify, BigCommerce)
- Cloud Storage (Dropbox, Google Drive, Box)
- FinTech (Stripe, PayPal, Square)
- Collaboration Tools (Slack, Microsoft Teams, Discord)
- Data Analytics (Tableau, Power BI, Qlik)
- CRM Software (Salesforce, HubSpot, Pipedrive)

### 🔹 Dynamic SWOT Analysis
- **Strengths:** Based on positive trends and achievements found in articles/tweets  
- **Weaknesses:** Extracted from critical mentions or issues  
- **Opportunities:** Derived from new features, funding, or expansions  
- **Threats:** Based on competition, regulation, or negative sentiment

### 🔹 Streamlit Dashboard
An interactive user interface to visualize:
- SWOT insights
- Real-time text summaries
- Competitor comparisons

---

## 📂 Project Structure

fda22/
│
├── streamlit_app.py # Main Streamlit application
├── COMPETITOR_ANALYSIS_ENHANCEMENT.md # Documentation of system improvements
├── requirements.txt # Python dependencies
├── test_competitor_analysis.py # Test for competitor mapping
├── test_enhanced_financial.py # Financial analysis validation
├── test_realtime.py # Real-time data fetching tests
├── test_swot.py # SWOT logic tests
├── .env # API credentials (Cohere, NewsAPI, Twitter)
└── venv/ # Virtual environment (excluded in .gitignore)

---

## 🧰 Technologies and Tools

| Category | Tool |
|-----------|------|
| **Frontend / UI** | Streamlit |
| **AI / NLP** | Cohere API |
| **Data Sources** | NewsAPI, Twitter API |
| **Programming Language** | Python |
| **Version Control** | Git & GitHub |
| **Testing** | Pytest / Unit Tests |
| **Visualization** | Streamlit Widgets |

---

## 🧠 Workflow Summary

| Step | Process |
|------|----------|
| 1️⃣ | User enters a company name (e.g., “Adobe” or “Zoom”) |
| 2️⃣ | Industry is auto-detected from context |
| 3️⃣ | Real competitors are fetched dynamically |
| 4️⃣ | Live data is collected from News & Twitter APIs |
| 5️⃣ | Cohere embeddings are used to analyze semantic similarity |
| 6️⃣ | SWOT analysis is generated using AI interpretation |
| 7️⃣ | Streamlit dashboard displays the results interactively |

---

## 📈 Learning Outcomes

Through this project, I learned to:
- Use **Git and GitHub** for structured, reproducible analytics workflows.  
- Integrate multiple APIs into a **real-time data pipeline**.  
- Apply **AI-based text embedding** and **cosine similarity** for contextual analysis.  
- Build an interactive **Streamlit dashboard** for business insights.  
- Manage collaboration and version control using **commits and branches** effectively.

---

## ✍️ Author

**Devadharshini S**  
Integrated M.Tech CSE (Business Analytics)  
VIT Chennai  
📧 dev.veldev@gmail.com  

---

## 🗂️ License

This project was developed for academic purposes as part of the  
**Foundation of Data Analytics (FDA22)** coursework at **VIT Chennai**.  
All datasets and APIs are used for educational and research demonstrations only.
