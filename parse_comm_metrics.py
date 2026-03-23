"""
Communication Metrics Parser for FL-REST
==========================================
Parses structured COMM_* log lines from simulation logs and produces:
  - Per-method aggregate: mean DL/UL bytes, transfer times
  - Per-round timeline: for time-to-accuracy plots  
  - Summary table: for the paper
  - Comparison across methods if multiple log files provided

Usage:
  python parse_comm_metrics.py fl_logs/simulation_*.log
  python parse_comm_metrics.py --compare fedprune.log fiarse.log fedrolex.log
  python parse_comm_metrics.py --round-metrics fl_logs/round_metrics.jsonl
"""

import re
import json
import argparse
import os
import sys
from collections import defaultdict


# =============================================================================
# Log Line Patterns
# =============================================================================

# COMM_DL: client=client_001 round=5 bytes=165432 timestamp=1234567890.1234
COMM_DL_PATTERN = re.compile(
    r"COMM_DL: client=(\S+) round=(\d+) bytes=(\d+) timestamp=([\d.]+)"
)

# COMM_UL: client=client_001 round=5 bytes=165432 timestamp=1234567890.1234
COMM_UL_PATTERN = re.compile(
    r"COMM_UL: client=(\S+) round=(\d+) bytes=(\d+) timestamp=([\d.]+)"
)

# COMM_DL_CLIENT: round=5 bytes=165432 transfer_sec=0.1234 throughput_kbps=1234.5
COMM_DL_CLIENT_PATTERN = re.compile(
    r"COMM_DL_CLIENT: round=(\d+) bytes=(\d+) transfer_sec=([\d.]+) throughput_kbps=([\d.]+)"
)

# COMM_UL_CLIENT: bytes=165432 transfer_sec=0.1234 throughput_kbps=1234.5
COMM_UL_CLIENT_PATTERN = re.compile(
    r"COMM_UL_CLIENT: bytes=(\d+) transfer_sec=([\d.]+) throughput_kbps=([\d.]+)"
)

# Payload size from existing logs: payload {X}KB / {Y}KB ({Z}%)
PAYLOAD_PATTERN = re.compile(
    r"(\w+): (\S+) payload ([\d.]+)KB / ([\d.]+)KB \(([\d.]+)%\)"
)

# Strategy method name from startup
METHOD_PATTERN = re.compile(
    r"(FedPrune|FIARSE|FLuID|HeteroFL|FedRolex)(?:: | initialized)"
)


def parse_log_file(filepath):
    """Parse a single simulation log file for communication metrics."""
    
    results = {
        "filepath": filepath,
        "method": None,
        "downloads": [],       # (client_id, round, bytes, timestamp)
        "uploads": [],         # (client_id, round, bytes, timestamp)
        "client_downloads": [], # (round, bytes, transfer_sec, throughput_kbps)
        "client_uploads": [],   # (bytes, transfer_sec, throughput_kbps)
        "payload_logs": [],     # (method, client, sub_kb, full_kb, pct)
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            # Method detection
            m = METHOD_PATTERN.search(line)
            if m and results["method"] is None:
                results["method"] = m.group(1)
            
            # Server-side download log
            m = COMM_DL_PATTERN.search(line)
            if m:
                results["downloads"].append({
                    "client": m.group(1),
                    "round": int(m.group(2)),
                    "bytes": int(m.group(3)),
                    "timestamp": float(m.group(4)),
                })
                continue
            
            # Server-side upload log
            m = COMM_UL_PATTERN.search(line)
            if m:
                results["uploads"].append({
                    "client": m.group(1),
                    "round": int(m.group(2)),
                    "bytes": int(m.group(3)),
                    "timestamp": float(m.group(4)),
                })
                continue
            
            # Client-side download timing
            m = COMM_DL_CLIENT_PATTERN.search(line)
            if m:
                results["client_downloads"].append({
                    "round": int(m.group(1)),
                    "bytes": int(m.group(2)),
                    "transfer_sec": float(m.group(3)),
                    "throughput_kbps": float(m.group(4)),
                })
                continue
            
            # Client-side upload timing
            m = COMM_UL_CLIENT_PATTERN.search(line)
            if m:
                results["client_uploads"].append({
                    "bytes": int(m.group(1)),
                    "transfer_sec": float(m.group(2)),
                    "throughput_kbps": float(m.group(3)),
                })
                continue
            
            # Payload size logs
            m = PAYLOAD_PATTERN.search(line)
            if m:
                results["payload_logs"].append({
                    "method": m.group(1),
                    "client": m.group(2),
                    "sub_kb": float(m.group(3)),
                    "full_kb": float(m.group(4)),
                    "pct": float(m.group(5)),
                })
    
    return results


def summarize(results):
    """Print summary statistics from parsed log."""
    method = results["method"] or "Unknown"
    print(f"\n{'='*60}")
    print(f"Method: {method}")
    print(f"Log: {results['filepath']}")
    print(f"{'='*60}")
    
    # Download stats
    if results["downloads"]:
        dl_bytes = [d["bytes"] for d in results["downloads"]]
        print(f"\nDownlink (server → client):")
        print(f"  Transfers: {len(dl_bytes)}")
        print(f"  Mean size: {sum(dl_bytes)/len(dl_bytes)/1024:.1f} KB")
        print(f"  Min:       {min(dl_bytes)/1024:.1f} KB")
        print(f"  Max:       {max(dl_bytes)/1024:.1f} KB")
        print(f"  Total:     {sum(dl_bytes)/1024/1024:.2f} MB")
    
    # Upload stats
    if results["uploads"]:
        ul_bytes = [u["bytes"] for u in results["uploads"]]
        print(f"\nUplink (client → server):")
        print(f"  Transfers: {len(ul_bytes)}")
        print(f"  Mean size: {sum(ul_bytes)/len(ul_bytes)/1024:.1f} KB")
        print(f"  Min:       {min(ul_bytes)/1024:.1f} KB")
        print(f"  Max:       {max(ul_bytes)/1024:.1f} KB")
        print(f"  Total:     {sum(ul_bytes)/1024/1024:.2f} MB")
    
    # Transfer timing stats (from client perspective)
    if results["client_downloads"]:
        dl_times = [d["transfer_sec"] for d in results["client_downloads"]]
        dl_tp = [d["throughput_kbps"] for d in results["client_downloads"]]
        print(f"\nDownload timing (client-measured):")
        print(f"  Mean transfer: {sum(dl_times)/len(dl_times):.3f}s")
        print(f"  Mean throughput: {sum(dl_tp)/len(dl_tp):.1f} kbps")
    
    if results["client_uploads"]:
        ul_times = [u["transfer_sec"] for u in results["client_uploads"]]
        ul_tp = [u["throughput_kbps"] for u in results["client_uploads"]]
        print(f"\nUpload timing (client-measured):")
        print(f"  Mean transfer: {sum(ul_times)/len(ul_times):.3f}s")
        print(f"  Mean throughput: {sum(ul_tp)/len(ul_tp):.1f} kbps")
    
    # Payload breakdown
    if results["payload_logs"]:
        print(f"\nPayload extraction log (first 5):")
        for p in results["payload_logs"][:5]:
            print(f"  {p['client']}: {p['sub_kb']:.1f}KB / {p['full_kb']:.1f}KB ({p['pct']:.1f}%)")


def compare_methods(log_files):
    """Compare communication metrics across multiple method logs."""
    all_results = []
    for f in log_files:
        if os.path.exists(f):
            all_results.append(parse_log_file(f))
        else:
            print(f"WARNING: {f} not found, skipping")
    
    if not all_results:
        print("No valid log files found.")
        return
    
    print(f"\n{'='*70}")
    print("COMMUNICATION COMPARISON")
    print(f"{'='*70}")
    
    header = f"{'Method':<12} {'Mean DL (KB)':<14} {'Mean UL (KB)':<14} {'Total/round (KB)':<18} {'DL time (s)':<12}"
    print(f"\n{header}")
    print("-" * len(header))
    
    for r in all_results:
        method = r["method"] or "?"
        
        dl_mean = 0
        if r["downloads"]:
            dl_mean = sum(d["bytes"] for d in r["downloads"]) / len(r["downloads"]) / 1024
        
        ul_mean = 0
        if r["uploads"]:
            ul_mean = sum(u["bytes"] for u in r["uploads"]) / len(r["uploads"]) / 1024
        
        dl_time = 0
        if r["client_downloads"]:
            dl_time = sum(d["transfer_sec"] for d in r["client_downloads"]) / len(r["client_downloads"])
        
        total = dl_mean + ul_mean
        print(f"{method:<12} {dl_mean:<14.1f} {ul_mean:<14.1f} {total:<18.1f} {dl_time:<12.4f}")


def parse_round_metrics(jsonl_path):
    """Parse the round_metrics.jsonl file for time-to-accuracy data."""
    rounds = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    
    if not rounds:
        print("No round metrics found.")
        return
    
    print(f"\n{'='*60}")
    print("ROUND METRICS (for time-to-accuracy plot)")
    print(f"{'='*60}")
    
    print(f"\n{'Round':<8} {'Accuracy':<12} {'Wall Time (s)':<15} {'Clients':<10}")
    print("-" * 45)
    
    t0 = rounds[0].get("wall_timestamp", 0)
    for r in rounds:
        rnd = r.get("round", "?")
        acc = r.get("accuracy", 0)
        wall = r.get("wall_timestamp", 0) - t0
        nc = r.get("n_clients", 0)
        print(f"{rnd:<8} {acc:<12.2f} {wall:<15.1f} {nc:<10}")


def main():
    parser = argparse.ArgumentParser(description="Parse FL-REST communication metrics")
    parser.add_argument("logs", nargs="*", help="Simulation log files to parse")
    parser.add_argument("--compare", action="store_true",
                        help="Compare metrics across multiple logs")
    parser.add_argument("--round-metrics", type=str, default=None,
                        help="Path to round_metrics.jsonl file")
    args = parser.parse_args()
    
    if args.round_metrics:
        parse_round_metrics(args.round_metrics)
    elif args.compare and args.logs:
        compare_methods(args.logs)
    elif args.logs:
        for log_path in args.logs:
            results = parse_log_file(log_path)
            summarize(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()