# Feedback Distributed Data-Gathering & Symmetries

## Overview

This project investigates feedback-based distributed data-gathering architectures and explores fundamental symmetry principles in physics. It combines **coding-based simulation** of feedback vs. non-feedback systems and **analytical proofs** in classical mechanics rooted in **Noether’s theorem**.

This Homework Set is from **Prof. Soojean Han's EE448F: Learning Patterns for Autonomous Control**, KAIST. 

---

## Problem 1: Feedback Distributed Data-Gathering — Simulations

### Objective:

Implement, simulate, and compare **Nonfeedback (NF)** and **Feedback (FB)** data-gathering architectures under more general conditions than those discussed in class.

### Key Modifications:

* Sampling period τ ≠ ∆t (now τ < ∆t)
* Smarter triggering rule for FB using latest available data
* Broadcasts from the central processor are **probabilistic**, not deterministic

### Simulation Parameters:

* **Environment**:

  * Step-size bounds: \[2.0, 4.0]
  * Step-size probability: 0.8
  * Period ∆t = 25.0

* **Sensors**:

  * Sampling period τ = 5.0
  * Noise variance σ² = 0.2
  * Offsets: b₁ = 0, b₂ = 4.0

* **Communication**:

  * Uplink delay ∆tᵤ = 2.0, power PU = 5.0
  * Downlink delay ∆t\_d = 1.0, power PD = 3.0

* **Central Processor**:

  * Initial estimate state 𝑥̂ ∈ Uniform\[−2, 2]

---

### Subparts:

* **(a)** Implement the full system architecture for both NF and FB based on the modified version of Example 2.
* **(b)** Simulate for **Tₛᵢₘ = 200.0**:

  * Plot transmission events over time for sensors and central processor.
  * Plot true vs. estimated trajectories for both NF and FB.
* **(c)** Run **50 Monte Carlo trials**:

  * Report **Mean Squared Error (MSE)**
  * Report **Average Power Expenditure**
* **(d)** Vary thresholds ϵ ∈ {0.9, 1.5, ..., 3.9}:

  * Plot MSE vs. Power trade-off curves.
  * Analyze which architecture performs better and when.
* **(e)** Vary broadcast probability p\_b ∈ {0.0, 0.2, ..., 1.0}:

  * Generate similar trade-off plots.
* **(f)** Increase system size:

  * ∆t = 40.0, state dimension n = 20, sensors M = 5
  * Sampling period τⱼ = 15.0, offsets bⱼ = 3(j−1)
  * Sensor coverage:

    * Sensor 1: \[9–12]
    * Sensor 2: \[5–12]
    * Sensor 3: \[9–16]
    * Sensor 4: \[1–8]
    * Sensor 5: \[13–20]
  * Analyze how shared components affect performance.
* **(g)** Propose **optimal strategies** in the FB architecture under this high-dimensional setting. Suggest additional parameters to modify for further improvements.

---

## Problem 2: Rotation Symmetry in Higher Dimensions

### Objective:

Extend the symmetry argument from 2D to 3D to demonstrate how **rotational symmetry** implies **conservation of angular momentum**.

### Key Concepts:

* Position: q = (x, y, z) ∈ ℝ³
* Use infinitesimal rotations to derive conserved quantities
* Analyze symmetry from a **Noether-theoretic** perspective

---

## Problem 3: Noether’s Theorem — Time Translation Symmetry

### Objective:

Show how **time-translation symmetry** implies **conservation of energy** using both Lagrangian and Hamiltonian mechanics.

---

### Subparts:

* **(a) Lagrangian Formulation**

  * Consider L(q, q̇) with q ∈ ℝⁿ
  * Apply time shift: ˜qᵢ(t) = qᵢ(t + ε(t))
  * Use Taylor expansion, integration by parts, and the principle of least action to prove conservation of energy

* **(b) Time-Dependent Lagrangian**

  * If L = L(t, q, q̇), is energy conserved?
  * Construct a counterexample (e.g., system under time-varying force Fₑₓₜ(t))

* **(c) Hamiltonian Formulation**

  * Consider H(p, q) in 1D
  * Rewrite action S = ∫ g(t) dt in terms of H
  * Use time-translation and Taylor expansion to show energy conservation
  * Leverage Hamilton’s equations to simplify

---

## Technologies Used

* **Python** (for simulation and plotting)
* **Control Systems Theory**
* **Distributed Estimation & Communication Models**
* **Classical Mechanics**
* **Lagrangian and Hamiltonian Mechanics**
* **Noether's Theorem**
* **Symmetry Analysis**

---

## Results Summary

* Developed a robust simulation comparing NF and FB data-gathering architectures
* Demonstrated how smarter feedback and probabilistic broadcasting can lead to lower power and better accuracy
* Verified fundamental physical principles: conservation of angular momentum and energy from symmetry
