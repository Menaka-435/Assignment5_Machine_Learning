# CS5710 – Machine Learning  
## Home Assignment 5  
### University of Central Missouri  

---

## 📌 Student Information
**Name:** Menaka Naga Sai Pothina  
**Course:** CS5710 – Machine Learning  
**Semester:** Fall 2025  
**Assignment:** Home Assignment 5    

---

## 📘 Overview
This repository contains my complete submission for **Home Assignment 5**, including:

- **Part A:** Short-answer responses (Transformer architecture, attention, ethics, dataset bias, harms, privacy, security)
- **Part B:** Full Python implementations for:
  - Scaled Dot-Product Attention (NumPy)
  - Simple Transformer Encoder Block (PyTorch)

All code is **fully commented**, structured clearly, and verified with sample input shapes.

---

---

## 🧠 Part A — Theory
Part A answers include concepts such as:

- Why positional encoding is required  
- How attention scores and softmax work  
- Multi-Head Attention benefits  
- Ethical foundations & AI harms  
- Dataset bias reasons  
- Privacy, data poisoning, and model stealing  

These answers are provided in the written submission.

---

## 🧩 Part B — Coding

### **1️⃣ Scaled Dot-Product Attention (NumPy)**  
File: `scaled_dot_product_attention.py`

This script implements:

- Dot-product attention score  
- Scaling by √dₖ  
- Softmax normalization  
- Context vector computation  

A built-in test prints:

- Q, K, V shapes  
- Context vector shape  
- Attention weights  

---

### **2️⃣ Simple Transformer Encoder Block (PyTorch)**  
File: `simple_transformer_encoder.py`

Includes:

- Manual Multi-Head Self-Attention  
- Residual connections (Add & Norm)  
- Feed-Forward Network (Linear → ReLU → Linear)  
- Layer Normalization  
- Output shape validation for:  
  - Batch size = 32  
  - Sequence length = 10  
  - Embedding dimension = 64  

---

## ⚙️ Installation & Running

### **Install dependencies**


