````markdown
# 🧠 Federated Learning Simulation Framework

[![Docker Build](https://img.shields.io/badge/Docker-Build-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red?logo=pytorch)](https://pytorch.org/)

---

A **comprehensive, containerized simulation framework for Federated Learning (FL)** — designed to run end-to-end FL experiments with **PyTorch**, simulating real-world conditions such as client dropouts, hardware differences, and network instability.  

The entire system runs inside **Docker containers** and is orchestrated through a single, simple **Bash master script**.

---

## 🚀 Features

- **Containerized & Scalable:** Uses **Docker** and **Docker Compose** to launch the server and multiple clients as isolated containers.  
- **Centralized Control:** A single `run.sh` script controls all simulation parameters.  
- **Real ML Pipeline:** Implements a **PyTorch CNN** trained on **CIFAR-10** using **Federated Averaging (FedAvg)**.  
- **Comprehensive Metrics:** Logs training time, payload size, and peak **RAM/GPU usage** for every client update.  
- **Realistic Simulations:** Built-in support for key FL challenges:  
  - 🧩 **Non-IID data:** Dirichlet distribution for client data heterogeneity.  
  - ⚙️ **Hardware heterogeneity:** Simulated “High-Perf” (GPU) and “Low-Perf” (CPU-limited) clients.  
  - ❌ **Client dropout:** Randomly removes clients mid-training.  
  - ⏱️ **Straggler handling:** Round timeout for slow clients.  
  - 🌐 **Network instability:** Simulates upload delays and download latency.  
- **Graceful Shutdown:** The server automatically saves the final model and terminates all containers cleanly, storing a timestamped log.

---

## 🛠️ Prerequisites

Before running, ensure your system has:

1. [**Docker**](https://docs.docker.com/get-docker/)
2. **Docker Compose** (usually included with Docker Desktop)
3. [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) *(optional, for GPU support)*
4. [**Git**](https://git-scm.com/downloads)

---

## ⚡ Quick Start

Run a complete simulation with just a few commands.  

> 💡 **Tip:** Don’t edit `config.py` or `docker-compose.yml` manually — use `run.sh` to configure everything.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Configure Simulation Parameters

Open `run.sh` and modify the parameters at the top:

```bash
#!/bin/bash
set -e

#################################################################
# 1. SIMULATION PARAMETERS
#################################################################

# --- Client & Round Config ---
CLIENTS_HIGH_PERF=2  # GPU-enabled clients
CLIENTS_LOW_PERF=3   # CPU-only clients
TOTAL_ROUNDS=10

# --- Aggregation Config ---
MIN_CLIENTS_FOR_AGGREGATION=3

# --- FL Condition Simulation ---
CLIENT_DROPOUT_RATE=0.1
ROUND_TIMEOUT_SEC=300
SLOW_SENDER_RATE=0.2
SLOW_SENDER_DELAY_SEC=30
NETWORK_LATENCY_RATE=0.0
NETWORK_LATENCY_DELAY_SEC=5

# --- Hardware Config ---
DEVICE="auto" # options: "auto", "cpu", "cuda"
```

### 3. Make the Script Executable

```bash
chmod +x run.sh
```

### 4. Run the Simulation

```bash
./run.sh
```

This script will:

1. Update `config.py` with your chosen parameters.
2. Auto-generate a `docker-compose.yml` file.
3. Build the container image.
4. Launch all containers (server + clients).
5. Stream live logs to your terminal and save them in `fl_logs/`.

---

## 📂 Project Structure

```
.
├── run.sh                  # 🚀 Master control script
├── config.py               # Central configuration (auto-modified)
├── generate_compose.py     # Generates docker-compose.yml
│
├── Dockerfile              # Defines the unified FL image
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores logs, datasets, etc.
│
├── server/
│   └── app.py              # Server logic, aggregation, and evaluation
│
└── client/
    ├── app.py              # Client logic (training & update submission)
    ├── model.py            # PyTorch model (SimpleCNN)
    ├── model_utils.py      # Serialization utilities
    └── trainer.py          # Local training and metrics collection
```

---

## ⚙️ Simulation Flow

1. You execute `./run.sh`.
2. The script updates `config.py` and creates `docker-compose.yml`.
3. `docker-compose up` builds the image and starts all containers.
4. The server initializes and waits for healthy clients.
5. Clients register, download the global model, train locally, and upload updates.
6. Aggregation occurs when either:

   * The quorum (`MIN_CLIENTS_FOR_AGGREGATION`) is met, **or**
   * The round timeout (`ROUND_TIMEOUT_SEC`) expires.
7. The server evaluates the new model, logs metrics, and starts the next round.
8. After all rounds, the final model is saved at:

   ```
   fl_logs/final_global_model.pth
   ```
9. A full run log is saved under:

   ```
   fl_logs/simulation_<timestamp>.log
   ```
10. The system shuts down gracefully, stopping all containers.

---

## 🧾 License

This project is released under the [MIT License](LICENSE).
Feel free to use, modify, and share it for your research or projects.

---

## 🤝 Contributing

Pull requests and issues are welcome!
If you’d like to contribute new features (e.g., new datasets or aggregation algorithms), please open an issue first to discuss design details.

---

## 🧩 Acknowledgments

* [PyTorch](https://pytorch.org/)
* [Docker](https://www.docker.com/)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

> ✨ Designed for reproducible Federated Learning research and educational simulation environments.

```
2. Generate a **`docs/` folder** template for GitHub Pages documentation (with setup + architecture diagrams)?
```
