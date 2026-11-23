from pathlib import Path
import numpy as np
import pandas as pd

def generate(path="data/sample_network_flows.csv", rows=1200, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["BENIGN", "DDoS", "PortScan"], size=rows, p=[0.65, 0.20, 0.15])
    attack = labels != "BENIGN"
    ddos = labels == "DDoS"
    n = rows
    df = pd.DataFrame({
        "flow_duration": np.where(attack, rng.gamma(2.0, 500, n), rng.gamma(3.0, 1800, n)),
        "total_fwd_packets": np.where(ddos, rng.poisson(80, n), rng.poisson(12, n)),
        "total_backward_packets": np.where(attack, rng.poisson(3, n), rng.poisson(15, n)),
        "flow_bytes_per_sec": np.where(ddos, rng.normal(90000, 15000, n), rng.normal(4000, 1200, n)),
        "flow_packets_per_sec": np.where(attack, rng.normal(1500, 300, n), rng.normal(25, 8, n)),
        "packet_length_mean": np.where(attack, rng.normal(250, 80, n), rng.normal(700, 100, n)),
        "packet_length_std": rng.gamma(2, 30, n),
        "syn_flag_count": np.where(attack, rng.poisson(5, n), rng.poisson(1, n)),
        "ack_flag_count": np.where(attack, rng.poisson(2, n), rng.poisson(10, n)),
        "destination_port": np.where(labels == "PortScan", rng.choice([21, 22, 23, 80, 443, 445, 3389], n),
                                     rng.choice([80, 443, 53, 8080], n)),
        "Label": labels
    })
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {path}")

if __name__ == "__main__":
    generate()
