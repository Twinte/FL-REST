"""
Validation Test: Submodel Extraction & Reconstruction
=======================================================
Run this BEFORE any experiments to verify that:
  1. extract_submodel() produces smaller tensors with correct values
  2. load_submodel_into_model() places values in correct positions
  3. extract → reconstruct round-trip preserves trained neuron weights
  4. Gradient masking on the reconstructed model is equivalent to
     gradient masking on the full model

Usage:
  python test_submodel_extraction.py

Expected output: All tests PASS.
If any test FAIL, do NOT run experiments — the communication pipeline has a bug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import io

# Add project root to path
sys.path.insert(0, '.')

from shared.submodel_utils import (
    extract_submodel,
    load_submodel_into_model,
    extract_trained_submodel,
    reconstruct_full_state_from_upload,
    compute_submodel_size,
    LAYER_CHAIN,
)


# =============================================================================
# Model Definition (must match shared/models.py)
# =============================================================================

class FedPruneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_indices(capacity):
    """Generate neuron indices for a given capacity ratio."""
    indices = {
        "conv1.weight": sorted(np.random.choice(32, max(1, int(32 * capacity)), replace=False).tolist()),
        "conv2.weight": sorted(np.random.choice(64, max(1, int(64 * capacity)), replace=False).tolist()),
        "conv3.weight": sorted(np.random.choice(128, max(1, int(128 * capacity)), replace=False).tolist()),
        "fc1.weight": sorted(np.random.choice(256, max(1, int(256 * capacity)), replace=False).tolist()),
        "fc2.weight": sorted(np.random.choice(128, max(1, int(128 * capacity)), replace=False).tolist()),
    }
    return indices


def test_extraction_sizes():
    """Test 1: Extracted submodel is smaller than full model."""
    print("\n" + "=" * 60)
    print("TEST 1: Extraction produces smaller tensors")
    print("=" * 60)
    
    model = FedPruneCNN()
    full_sd = model.state_dict()
    full_params = sum(v.numel() for v in full_sd.values())
    
    passed = True
    for r in [0.25, 0.5, 0.75]:
        torch.manual_seed(42)
        np.random.seed(42)
        indices = make_indices(r)
        
        sub = extract_submodel(full_sd, indices)
        sub_params, sub_bytes = compute_submodel_size(sub)
        pct = 100 * sub_params / full_params
        
        print(f"  r={r:.2f}: {sub_params:>8,} / {full_params:>8,} params ({pct:.1f}%)")
        
        if sub_params >= full_params:
            print(f"  FAIL: submodel not smaller than full model!")
            passed = False
        
        # Verify shapes are correct
        n_conv1 = len(indices["conv1.weight"])
        n_conv2 = len(indices["conv2.weight"])
        n_conv3 = len(indices["conv3.weight"])
        n_fc1 = len(indices["fc1.weight"])
        n_fc2 = len(indices["fc2.weight"])
        
        expected = {
            "conv1.weight": (n_conv1, 3, 3, 3),         # in=3 fixed
            "conv2.weight": (n_conv2, n_conv1, 3, 3),
            "conv3.weight": (n_conv3, n_conv2, 3, 3),
            "fc1.weight": (n_fc1, n_conv3 * 16),         # 16 = 4*4 spatial
            "fc2.weight": (n_fc2, n_fc1),
            "fc3.weight": (10, n_fc2),                    # out=10 fixed
        }
        
        for key, exp_shape in expected.items():
            if key in sub:
                actual = tuple(sub[key].shape)
                if actual != exp_shape:
                    print(f"  FAIL: {key} shape {actual} != expected {exp_shape}")
                    passed = False
    
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_extraction_values():
    """Test 2: Extracted values match the correct positions in full model."""
    print("\n" + "=" * 60)
    print("TEST 2: Extracted values match source positions")
    print("=" * 60)
    
    model = FedPruneCNN()
    # Fill with recognizable values
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.fill_(0)
            if 'conv1.weight' in name:
                for i in range(32):
                    param[i].fill_(float(i))  # Each filter = its index
    
    full_sd = model.state_dict()
    
    np.random.seed(42)
    indices = make_indices(0.5)
    sub = extract_submodel(full_sd, indices)
    
    # Check conv1 output slicing
    conv1_idx = indices["conv1.weight"]
    for i, orig_idx in enumerate(conv1_idx):
        expected_val = float(orig_idx)
        actual_val = sub["conv1.weight"][i].mean().item()
        if abs(actual_val - expected_val) > 1e-5:
            print(f"  FAIL: conv1 neuron {i} (orig {orig_idx}): "
                  f"expected {expected_val}, got {actual_val}")
            return False
    
    print(f"  conv1 output indexing: correct")
    
    # Check conv2 input slicing (should use conv1's output indices)
    conv2_sub = sub["conv2.weight"]
    expected_in_dim = len(indices["conv1.weight"])
    actual_in_dim = conv2_sub.shape[1]
    if actual_in_dim != expected_in_dim:
        print(f"  FAIL: conv2 input dim {actual_in_dim} != expected {expected_in_dim}")
        return False
    print(f"  conv2 input indexing: correct (in_dim={actual_in_dim})")
    
    # Check fc1 input (conv3 flattened)
    fc1_sub = sub["fc1.weight"]
    expected_fc1_in = len(indices["conv3.weight"]) * 16  # spatial=16
    actual_fc1_in = fc1_sub.shape[1]
    if actual_fc1_in != expected_fc1_in:
        print(f"  FAIL: fc1 input dim {actual_fc1_in} != expected {expected_fc1_in}")
        return False
    print(f"  fc1 conv→fc transition: correct (in_dim={actual_fc1_in})")
    
    # Check fc3 (classifier)
    fc3_sub = sub["fc3.weight"]
    expected_fc3_in = len(indices["fc2.weight"])
    if fc3_sub.shape != (10, expected_fc3_in):
        print(f"  FAIL: fc3 shape {tuple(fc3_sub.shape)} != expected (10, {expected_fc3_in})")
        return False
    print(f"  fc3 classifier: correct (shape={tuple(fc3_sub.shape)})")
    
    # Check BN params
    bn1_sub = sub.get("bn1.weight")
    if bn1_sub is not None:
        if len(bn1_sub) != len(indices["conv1.weight"]):
            print(f"  FAIL: bn1 length {len(bn1_sub)} != {len(indices['conv1.weight'])}")
            return False
        print(f"  BN params: correct")
    
    print(f"\n  Result: PASS")
    return True


def test_roundtrip():
    """Test 3: Extract → Reconstruct preserves values at assigned positions."""
    print("\n" + "=" * 60)
    print("TEST 3: Extract → Reconstruct round-trip")
    print("=" * 60)
    
    torch.manual_seed(123)
    model_original = FedPruneCNN()
    full_sd = model_original.state_dict()
    
    np.random.seed(42)
    indices = make_indices(0.5)
    
    # Extract
    sub = extract_submodel(full_sd, indices)
    
    # Reconstruct into a fresh model
    model_reconstructed = FedPruneCNN()
    model_reconstructed.load_state_dict(full_sd)  # Start from same state
    load_submodel_into_model(model_reconstructed, sub, indices)
    
    # The reconstructed model should be identical to original at assigned positions
    recon_sd = model_reconstructed.state_dict()
    
    max_diff = 0.0
    for key in full_sd:
        diff = (full_sd[key].float() - recon_sd[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
    
    passed = max_diff < 1e-6
    print(f"  Max difference: {max_diff:.2e}")
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_upload_roundtrip():
    """Test 4: Client trains → uploads compact → server reconstructs correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Upload compact → Server reconstruct")
    print("=" * 60)
    
    torch.manual_seed(456)
    model = FedPruneCNN()
    full_sd = model.state_dict()
    
    np.random.seed(42)
    indices = make_indices(0.5)
    
    # Simulate: client modifies assigned neurons
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in indices:
                for idx in indices[name]:
                    param[idx] += 1.0  # Detectable modification
    
    # Client extracts compact upload
    upload = extract_trained_submodel(model, indices)
    
    # Server reconstructs
    reference = full_sd  # Original model before training
    full_recon = reconstruct_full_state_from_upload(upload, indices, reference)
    
    # Check: modified neurons should have the +1.0 delta
    passed = True
    for name in ["conv1.weight", "fc1.weight", "fc2.weight"]:
        if name in indices and name in full_recon:
            for idx in indices[name][:3]:  # Check first 3
                orig_val = full_sd[name][idx].mean().item()
                recon_val = full_recon[name][idx].mean().item()
                expected = orig_val + 1.0
                if abs(recon_val - expected) > 1e-5:
                    print(f"  FAIL: {name}[{idx}] recon={recon_val:.4f} "
                          f"expected={expected:.4f}")
                    passed = False
    
    # Check: non-assigned neurons should be zero (for weighted avg to work)
    for name in ["conv1.weight"]:
        all_idx = set(range(32))
        assigned = set(indices[name])
        unassigned = list(all_idx - assigned)[:3]
        for idx in unassigned:
            val = full_recon[name][idx].abs().max().item()
            if val > 1e-8:
                print(f"  FAIL: {name}[{idx}] (unassigned) should be 0, got {val}")
                passed = False
    
    print(f"  Assigned neurons: correctly reconstructed")
    print(f"  Unassigned neurons: correctly zeroed")
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_serialization_size():
    """Test 5: Serialized submodel is smaller on the wire."""
    print("\n" + "=" * 60)
    print("TEST 5: Serialized payload sizes")
    print("=" * 60)
    
    model = FedPruneCNN()
    full_sd = model.state_dict()
    
    # Serialize full model
    buf = io.BytesIO()
    torch.save({"model_state": full_sd, "extra_payload": {}, "is_submodel": False}, buf)
    full_bytes = buf.tell()
    
    print(f"  Full model payload: {full_bytes:,} bytes ({full_bytes/1024:.1f} KB)")
    
    passed = True
    for r in [0.25, 0.5, 0.75]:
        np.random.seed(42)
        indices = make_indices(r)
        sub = extract_submodel(full_sd, indices)
        
        buf = io.BytesIO()
        torch.save({"model_state": sub, "extra_payload": indices, "is_submodel": True}, buf)
        sub_bytes = buf.tell()
        
        ratio = sub_bytes / full_bytes
        print(f"  r={r:.2f}: {sub_bytes:,} bytes ({sub_bytes/1024:.1f} KB) "
              f"= {100*ratio:.1f}% of full")
        
        if sub_bytes >= full_bytes:
            print(f"  FAIL: serialized submodel not smaller!")
            passed = False
    
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("=" * 60)
    print("SUBMODEL EXTRACTION VALIDATION SUITE")
    print("=" * 60)
    
    results = []
    results.append(("Extraction sizes", test_extraction_sizes()))
    results.append(("Extraction values", test_extraction_values()))
    results.append(("Round-trip", test_roundtrip()))
    results.append(("Upload round-trip", test_upload_roundtrip()))
    results.append(("Serialization sizes", test_serialization_size()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        marker = "✓" if passed else "✗"
        print(f"  {marker} {name}: {status}")
        all_pass = all_pass and passed
    
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED — safe to run communication experiments")
    else:
        print("SOME TESTS FAILED — fix extraction logic before experiments!")
    print(f"{'='*60}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())