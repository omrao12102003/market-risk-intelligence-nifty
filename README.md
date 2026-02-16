Subject: README.md – Market Risk Intelligence & Regime Detection System

# Market Risk Intelligence & Regime Detection System (NIFTY 50)

Built a market risk intelligence system for NIFTY 50 using rule-based logic and Hidden Markov Models to detect market regimes and control exposure, improving Sharpe ratio and reducing maximum drawdown versus buy-and-hold.

## 1. Introduction

This project builds a **market risk intelligence system** for the NIFTY 50 index.

The goal is **not price prediction**.
Instead, the system focuses on **understanding market regimes**, validating them financially, and **using regimes to control risk and exposure**.

The project is designed in a **step-by-step, industry-style workflow**, progressing from raw data to regime-aware strategy evaluation.

A new user can follow this project **from Day 0 to Day 9** and fully reconstruct both the logic and the code.

---

## 2. Core Idea

Financial markets are **non-stationary**.
Returns are noisy and difficult to forecast reliably.

However:

* Volatility clusters
* Drawdowns cluster
* Market behavior changes in regimes

This project is built on the principle:

> It is more robust to **adapt exposure based on market regime** than to predict future prices.

---

## 3. Project Structure

```
market-risk-intelligence-nifty/
│
├── main.py                     # Pipeline controller (runs all days)
├── risk_engine/                # Modular analytics logic
│   ├── regime_detection.py     # Rule-based regime logic
│   ├── regime_analysis.py      # Regime persistence & transitions
│   ├── regime_performance.py   # Regime-wise performance
│   ├── hmm_regime.py           # HMM regime detection (ML)
│   ├── hmm_analysis.py         # Interpretation of HMM states
│   ├── strategy_overlay.py     # Regime-aware strategies
│   └── performance_metrics.py # Sharpe & drawdown calculations
│
├── data/
│   └── nifty50_daily.csv       # Historical NIFTY 50 data
│
├── results/                    # Generated CSV outputs (not versioned)
└── README.md
```

Design principles:

* `main.py` controls the full pipeline
* All reusable logic lives inside `risk_engine/`
* Outputs are generated as CSVs
* No notebook dependency

---





---

Environment & Setup

* Created virtual environment
* Installed required libraries
* Initialized Git repository
* Defined clean folder structure

Purpose:

* Ensure reproducibility
* Follow professional Python project practices

---

Data Ingestion & Pipeline Structure

What was done:

* Loaded NIFTY 50 daily OHLCV data
* Standardized column names
* Converted date column safely
* Set up `main.py` as the single entry point

Why:

* A clean data pipeline is the foundation of any quant system
* Avoids notebook-only, non-reproducible workflows

---

Risk Metrics Computation

Computed:

* Daily returns
* Rolling 20-day annualized volatility
* Cumulative returns
* Drawdowns

Why:

* Markets are defined by **risk behavior**, not just returns
* These metrics form the **inputs to regime detection**

---

Rule-Based Regime Detection

Built:

* Volatility-based classification (low / medium / high)
* Drawdown thresholds
* Combined logic to label:

  * RISK-ON
  * NEUTRAL
  * RISK-OFF

Output:

* Daily regime labels

Why:

* Convert raw risk metrics into **interpretable market states**
* Provide a transparent baseline before ML

---

Regime Persistence & Transitions

Analyzed:

* How long regimes persist
* How often regimes switch
* Transition probabilities between regimes

Why:

* Real market regimes cluster
* Random switching would invalidate regime logic

This step validates **regime stability**.

---

Regime Performance Validation

Computed (per regime):

* Average and annualized returns
* Volatility
* Sharpe ratio
* Maximum drawdown
* Hit ratio

Why:

* A regime must be **economically meaningful**
* Statistical regimes without performance differences are useless

---

Hidden Markov Model (HMM) Regime Detection

Introduced:

* Unsupervised machine learning
* Gaussian Hidden Markov Model
* Features:

  * Returns
  * Volatility
  * Drawdown

Output:

* Latent market states discovered by the model

Why:

* Remove human-defined thresholds
* Allow data-driven regime discovery

---

HMM Regime Interpretation

Performed:

* Financial analysis of each HMM state
* Sharpe, volatility, and drawdown per state
* Mapping ML states to economic meaning

Why:

* ML outputs are meaningless without financial interpretation
* This step validates whether HMM regimes are sensible

---

Regime-Aware Strategy Overlay

Built strategies:

1. Buy & Hold (baseline)
2. Rule-based regime strategy
3. HMM-based regime strategy

Logic:

* Full exposure in favorable regimes
* Reduced or zero exposure in risky regimes

Why:

* Regimes are useful only if they **change decisions**
* This introduces a real portfolio perspective

---

Strategy Performance & Stress Testing

Evaluated:

* Equity curves
* Sharpe ratio
* Maximum drawdown

Compared:

* Buy & Hold vs regime-aware strategies

Findings:

* Regime-aware strategies significantly reduce drawdowns
* Risk-adjusted performance improves

---



CSV Outputs and Their Purpose

Each CSV represents a **specific stage of analysis**:

* Daily regime labels
* Regime persistence and transitions
* Regime-wise economic validation
* HMM-discovered regimes
* Financial interpretation of HMM states
* Strategy daily returns
* Final performance comparison

These files allow:

* Independent validation
* Reproducibility
* Further research or extensions

---

What This Project Does and Does Not Do

### This project does:

* Detect market regimes
* Validate regimes financially
* Control exposure using regimes
* Improve risk-adjusted outcomes

This project does not:

* Predict prices
* Guarantee profits
* Include transaction costs or execution modeling

---

 How to Run the Project

1. Activate virtual environment
2. Place data in `data/`
3. Run:

   ```
   python main.py
   ```
4. Outputs are generated in `results/`

---

## 8. Final Summary

This project demonstrates:

* Risk-focused market thinking
* Regime-based analysis
* Responsible use of machine learning
* Clean, modular system design



---



