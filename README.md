# email_spam
# Email Spam Classifier (MLOps Practice Project)

This is a **practice project** created while learning **MLOps** concepts such as data versioning, pipeline creation, and experiment tracking using **DVC** and **Git**.

---

##  Goal

Build a simple machine learning pipeline to classify emails as spam or not spam, and integrate MLOps tools like:

- ✅ DVC (for tracking data, models, and experiments)
- ✅ Git (for version control)
- ✅ params.yaml (to manage configurations)

---

## 📂 Project Structure
email_spam/
├── data/ # Data files (managed by DVC)
├── models/ # Saved ML models (tracked with DVC)
├── src/ # Scripts for training, prediction, etc.
├── params.yaml # Parameters for training
├── dvc.yaml # DVC pipeline stages
├── dvc.lock # Auto-generated DVC lock file
├── .gitignore
├── .dvcignore
└── README.md # This file

---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sayyedsabirali/email_spam.git
cd email_spam

