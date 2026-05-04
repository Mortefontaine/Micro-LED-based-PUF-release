import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hashlib
from reedsolo import RSCodec

# ====================== Parameters ======================
SEGMENT_COUNT = 4
TOTAL_BITS = 256
SEGMENT_BITS = TOTAL_BITS // SEGMENT_COUNT      # 64
DATA_BYTES_PER_SEG = SEGMENT_BITS // 8          # 8
NSYM_PER_SEG = 16                               # RS parity bytes
AUTH_KEY_BITS = 128

# ====================== Helpers ======================
def bin_array_to_bytes(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    return np.packbits(arr).tobytes()

def bytes_to_bin_array(b, length_bits):
    arr = np.unpackbits(np.frombuffer(bytes(b), dtype=np.uint8))
    return arr[:length_bits].astype(np.uint8)

def hamming_bits(a, b):
    return int(np.sum(np.asarray(a, dtype=np.uint8) != np.asarray(b, dtype=np.uint8)))

def derive_auth_key_128(bits):
    """
    Cryptographic hash-based key derivation / privacy amplification.
    Input : stabilized 256-bit PUF response
    Output: conservative 128-bit authentication key
    """
    bits = np.asarray(bits, dtype=np.uint8)
    raw_bytes = np.packbits(bits).tobytes()
    digest = hashlib.sha256(b"microLED-PUF-auth-v1|" + raw_bytes).digest()
    key_bytes = digest[:AUTH_KEY_BITS // 8]
    return np.unpackbits(np.frombuffer(key_bytes, dtype=np.uint8)).astype(np.uint8)

def bits_to_hex(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()

# ====================== Load data ======================
embeddings = np.load("binary_embedding_results/binarized_test_embeddings.npy").astype(np.uint8)
labels = np.load("binary_embedding_results/corresponding_test_labels.npy")

chip_ids = sorted(np.unique(labels))
num_chips = len(chip_ids)
chip_labels = {chip_ids[i]: f"M{i+1}" for i in range(num_chips)}
print("Found chips:", [chip_labels[c] for c in chip_ids])

# ====================== Output folder ======================
os.makedirs("PUF_results", exist_ok=True)

# ====================== Enrollment & reconstruction ======================
rsc = RSCodec(NSYM_PER_SEG)

raw_blocks = []
enrolled_responses = []
auth_keys_128 = []
helper_records = []
reconstruction_records = []

for c in chip_ids:
    idxs = np.where(labels == c)[0]
    chip_data = embeddings[idxs].astype(np.uint8)
    raw_blocks.append(chip_data)

    chip_name = chip_labels[c]
    enroll_bits = chip_data[0].astype(np.uint8)
    enrolled_key_128 = derive_auth_key_128(enroll_bits)

    enrolled_responses.append(enroll_bits)
    auth_keys_128.append(enrolled_key_128)

    # ----- Enrollment: store only public RS parity bytes as helper data -----
    parity_helpers = []

    for s in range(SEGMENT_COUNT):
        seg_bits = enroll_bits[s * SEGMENT_BITS:(s + 1) * SEGMENT_BITS]
        seg_bytes = bin_array_to_bytes(seg_bits)

        encoded = bytes(rsc.encode(seg_bytes))  # systematic RS: message || parity
        parity = encoded[DATA_BYTES_PER_SEG:]   # public helper data
        parity_helpers.append(parity)

        helper_records.append({
            "chip": chip_name,
            "segment": s,
            "rs_code": f"RS({DATA_BYTES_PER_SEG + NSYM_PER_SEG},{DATA_BYTES_PER_SEG})",
            "public_parity_hex": parity.hex()
        })

    # ----- Reconstruction evaluation -----
    total_tests = max(0, chip_data.shape[0] - 1)
    success = 0
    input_bers = []
    residual_bers = []

    for j in range(1, chip_data.shape[0]):
        sample_bits = chip_data[j].astype(np.uint8)
        input_err = hamming_bits(sample_bits, enroll_bits)
        input_ber = input_err / TOTAL_BITS
        input_bers.append(input_ber)

        recovered_segments = []
        seg_failed = False

        for s in range(SEGMENT_COUNT):
            seg_bits_noisy = sample_bits[s * SEGMENT_BITS:(s + 1) * SEGMENT_BITS]
            seg_bytes_noisy = bin_array_to_bytes(seg_bits_noisy)

            # Reconstruct systematic RS codeword using noisy message + public parity
            recovered_codeword = seg_bytes_noisy + parity_helpers[s]

            try:
                dec = rsc.decode(recovered_codeword)
                msg = dec[0] if isinstance(dec, tuple) else dec
                recovered_bits = bytes_to_bin_array(msg, SEGMENT_BITS)
            except Exception:
                seg_failed = True
                recovered_bits = np.zeros(SEGMENT_BITS, dtype=np.uint8)

            recovered_segments.append(recovered_bits)

        recovered_full = np.hstack(recovered_segments).astype(np.uint8)
        residual_err = hamming_bits(recovered_full, enroll_bits)
        residual_ber = residual_err / TOTAL_BITS
        residual_bers.append(residual_ber)

        reconstructed_key_128 = derive_auth_key_128(recovered_full)
        key_match = np.array_equal(reconstructed_key_128, enrolled_key_128)

        recon_success = (residual_err == 0) and (not seg_failed) and key_match
        if recon_success:
            success += 1

        reconstruction_records.append({
            "chip": chip_name,
            "sample_index": int(j),
            "input_bit_errors": input_err,
            "input_ber": input_ber,
            "residual_bit_errors_after_fe": residual_err,
            "residual_ber_after_fe": residual_ber,
            "reconstruction_success": recon_success,
            "key_match_128": key_match
        })

    success_rate = (success / total_tests * 100) if total_tests > 0 else 100.0
    print(
        f"{chip_name} | success = {success_rate:.2f}% | "
        f"mean input BER = {np.mean(input_bers):.4f} | "
        f"max input BER = {np.max(input_bers):.4f}"
    )

# ====================== Merge matrices ======================
raw_all = np.vstack(raw_blocks)
enrolled_matrix = np.vstack(enrolled_responses)
auth_key_matrix = np.vstack(auth_keys_128)

# ====================== Save outputs ======================
pd.DataFrame(raw_all).to_csv("PUF_results/merged_raw_bits.csv", index=False, header=False)
pd.DataFrame(enrolled_matrix).to_csv("PUF_results/enrolled_256bit_puf_responses.csv", index=False, header=False)
pd.DataFrame(auth_key_matrix).to_csv("PUF_results/derived_128bit_auth_keys.csv", index=False, header=False)

pd.DataFrame({
    "chip": [chip_labels[c] for c in chip_ids],
    "auth_key_128_hex": [bits_to_hex(k) for k in auth_key_matrix]
}).to_csv("PUF_results/derived_128bit_auth_keys_hex.csv", index=False)

pd.DataFrame(helper_records).to_csv("PUF_results/public_helper_data_rs_parity.csv", index=False)
pd.DataFrame(reconstruction_records).to_csv("PUF_results/reconstruction_results.csv", index=False)

# ====================== Bit balance sanity check ======================
print("\nAverage bit probability of 128-bit auth keys:", auth_key_matrix.mean())

# ====================== Visualization ======================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(raw_all, aspect="auto", cmap="gray_r")
plt.title("Raw 256-bit PUF Responses")
plt.xlabel("Bit index")
plt.ylabel("Sample index")
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(auth_key_matrix, aspect="auto", cmap="gray_r")
plt.title("Derived 128-bit Authentication Keys")
plt.xlabel("Bit index")
plt.ylabel("Chip")
plt.yticks(range(num_chips), [chip_labels[c] for c in chip_ids])

plt.tight_layout()
plt.savefig("PUF_results/puf_fuzzy_extractor_128bit_keys.png", dpi=300)
plt.show()
