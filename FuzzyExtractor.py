import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hashlib
from reedsolo import RSCodec

# ====================== Parameters ======================
SEGMENT_COUNT = 4
SEGMENT_BITS = 256 // SEGMENT_COUNT        # 64
DATA_BYTES_PER_SEG = SEGMENT_BITS // 8     # 8
NSYM_PER_SEG = 16                          # RS redundancy bytes
TOTAL_BITS = 256

# ====================== Helpers ======================
def bin_array_to_bytes(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    return np.packbits(arr).tobytes()

def bytes_to_bin_array(b, length_bits):
    arr = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return arr[:length_bits].astype(np.uint8)

def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

# ---------- Hash-based randomness extractor ----------
def hash_extractor(bits: np.ndarray) -> np.ndarray:
    """
    SHA-256 based randomness extractor
    Input : 256-bit stable (but possibly biased) PUF key
    Output: 256-bit uniformly distributed key
    """
    bits = bits.astype(np.uint8)
    raw_bytes = np.packbits(bits).tobytes()
    digest = hashlib.sha256(raw_bytes).digest()   # 32 bytes = 256 bits
    return np.unpackbits(
        np.frombuffer(digest, dtype=np.uint8)
    ).astype(np.uint8)

# ====================== Load data ======================
embeddings = np.load("binary_embedding_results/binarized_test_embeddings.npy")
labels = np.load("binary_embedding_results/corresponding_test_labels.npy")

embeddings = embeddings.astype(np.uint8)
labels = np.asarray(labels)

chip_ids = sorted(np.unique(labels))
num_chips = len(chip_ids)
chip_labels = {chip_ids[i]: f"M{i+1}" for i in range(num_chips)}
print("Found chips:", [chip_labels[c] for c in chip_ids])

# ====================== Output folder ======================
os.makedirs("PUF_results", exist_ok=True)

# ====================== FE enrollment & reconstruction ======================
stable_keys = []          # final 256-bit extracted keys (after hash)
per_chip_stats = {}
raw_blocks = []

rsc = RSCodec(NSYM_PER_SEG)

for c in chip_ids:
    idxs = np.where(labels == c)[0]
    if len(idxs) == 0:
        continue

    chip_data = embeddings[idxs]
    raw_blocks.append(chip_data)

    enroll_bits = chip_data[0].astype(np.uint8)

    # ----- Build helper data -----
    helpers = []
    encoded_lengths = []

    for s in range(SEGMENT_COUNT):
        seg_bits = enroll_bits[s*SEGMENT_BITS:(s+1)*SEGMENT_BITS]
        seg_bytes = bin_array_to_bytes(seg_bits)
        encoded = rsc.encode(seg_bytes)
        nlen = len(encoded)

        encoded_lengths.append(nlen)
        padded = seg_bytes + b'\x00' * (nlen - len(seg_bytes))
        helper = xor_bytes(encoded, padded)
        helpers.append(helper)

    # ----- Reconstruction evaluation -----
    total_tests = max(0, chip_data.shape[0] - 1)
    success = 0
    residual_errors = []

    for j in range(1, chip_data.shape[0]):
        sample_bits = chip_data[j].astype(np.uint8)
        recovered_segments = []
        seg_failed = False

        for s in range(SEGMENT_COUNT):
            seg_bits_noisy = sample_bits[s*SEGMENT_BITS:(s+1)*SEGMENT_BITS]
            seg_bytes_noisy = bin_array_to_bytes(seg_bits_noisy)

            nlen = encoded_lengths[s]
            padded_noisy = seg_bytes_noisy + b'\x00' * (nlen - len(seg_bytes_noisy))
            recovered_codeword = xor_bytes(padded_noisy, helpers[s])

            try:
                dec = rsc.decode(recovered_codeword)
                msg = dec[0] if isinstance(dec, tuple) else dec
                recovered_bits = bytes_to_bin_array(msg, SEGMENT_BITS)
                recovered_segments.append(recovered_bits)
            except Exception:
                seg_failed = True
                recovered_segments.append(np.zeros(SEGMENT_BITS, dtype=np.uint8))

        recovered_full = np.hstack(recovered_segments)
        err = hamming_bits(recovered_full, enroll_bits)
        residual_errors.append(err)

        if err == 0 and not seg_failed:
            success += 1

    # ----- Apply randomness extractor (SHA-256) -----
    extracted_key = hash_extractor(enroll_bits)
    stable_keys.append(extracted_key)

    avg_err = np.mean(residual_errors) if residual_errors else 0.0
    success_rate = (success / total_tests * 100) if total_tests > 0 else 100.0

    per_chip_stats[chip_labels[c]] = {
        "success_rate_pct": success_rate,
        "avg_residual_errors": avg_err,
        "num_tests": total_tests
    }

    print(f"{chip_labels[c]} | success = {success_rate:.2f}% | avg err = {avg_err:.2f}")

# ====================== Merge matrices ======================
raw_all = np.vstack(raw_blocks)
stable_matrix = np.vstack(stable_keys)

# ====================== Save CSVs ======================
pd.DataFrame(raw_all).to_csv("PUF_results/merged_raw_bits.csv", index=False, header=False)
pd.DataFrame(stable_matrix).to_csv("PUF_results/merged_stable_bits.csv", index=False, header=False)

# ====================== Bit balance sanity check ======================
bit_prob = stable_matrix.mean()
print("\nAverage bit probability (should be ~0.5):", bit_prob)

# ====================== Visualization ======================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(raw_all, aspect='auto', cmap='gray_r')
plt.title("Raw PUF Measurements")
plt.xlabel("Bit index")
plt.ylabel("Sample index")
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(stable_matrix, aspect='auto', cmap='gray_r')
plt.title("Extracted Stable Keys (SHA-256)")
plt.xlabel("Bit index")
plt.ylabel("Chip")

ax = plt.gca()
ax.set_yticks(range(num_chips))
ax.set_yticklabels([chip_labels[c] for c in chip_ids])


plt.tight_layout()
plt.show()

