# 🛍️ Customer Segmentation Analysis Dashboard

A production-ready **Customer Segmentation System** built using Machine Learning and deployed with an interactive **Streamlit dashboard**. This project segments customers into meaningful groups and maps them into business-friendly personas such as *VIP, Budget, Premium*, etc.

---

## 🚀 Project Overview

Customer Segmentation is a core concept in **marketing analytics** and **data science**, used to divide customers into groups based on shared characteristics.

This project leverages:

* **K-Means Clustering**
* **Feature Engineering (Age + Gender Encoding)**
* **Real-Time Prediction System**
* **Interactive Dashboard (Streamlit)**

---

## 🎯 Key Features

✔️ Advanced clustering using multiple features
✔️ Gender encoding & data preprocessing
✔️ Real-time customer input & prediction
✔️ Customer persona classification (VIP, Budget, etc.)
✔️ Interactive data visualization
✔️ Clean and user-friendly UI

---

## 🧠 Machine Learning Workflow

1. Data Collection (Mall Customer Dataset)
2. Data Preprocessing

   * Handling categorical data (Gender Encoding)
   * Feature Scaling (StandardScaler)
3. Model Training

   * K-Means Clustering
4. Cluster Analysis
5. Persona Mapping
6. Deployment with Streamlit

---

## 📊 Features Used

* Gender (Encoded)
* Age
* Annual Income (k$)
* Spending Score (1–100)

---

## 🧩 Customer Personas

| Cluster | Persona              | Description                    |
| ------- | -------------------- | ------------------------------ |
| 0       | 💰 Budget Customer   | Low income, low spending       |
| 1       | 🙂 Standard Customer | Average behavior               |
| 2       | 💎 Premium Customer  | High income, moderate spending |
| 3       | 🔥 High Spender      | High spending customers        |
| 4       | 🧠 Careful Saver     | High income but low spending   |

---

## 🖥️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib / Seaborn**
* **Streamlit**

---

## 📁 Project Structure

```
Customer-Segmentation/
│── app.py
│── model.py
│── Mall_Customers.csv
│── segmented_customers.csv
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

> Add your dashboard screenshots here
> Example:

```
![Dashboard](assets/dashboard.png)
```

---

## 📈 Output

* Clustered customer dataset
* Real-time persona prediction
* Visual segmentation graphs

---

## 💼 Business Use Cases

* Targeted Marketing Campaigns
* Customer Retention Strategies
* Personalized Recommendations
* Revenue Optimization

---

## 🧪 Future Enhancements

* 🔗 Integration with real-time database (Firebase/MySQL)
* 📊 Advanced visualizations using Plotly
* 🔐 Authentication system
* ☁️ Deployment on cloud platforms
* 🤖 Use of advanced ML models (DBSCAN, Hierarchical Clustering)


---

## 👨‍💻 Author

**Anand Mahato**
B.Tech CSE | Data Science Enthusiast

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and share it with others!
