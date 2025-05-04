import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

raw = np.fromfile('raw_audio_u8_44k.raw', dtype=np.uint8)
offset = 10000
N_BITS = 13_000_000
samples_needed = int(np.ceil(N_BITS / 3))
raw = raw[offset:offset + samples_needed]

lsb3 = raw & 0b00000111
# save raw 3 LSB values as uint8 file
lsb3.tofile('data.u8')
# extract bits (LSB first per sample)
bit_stream = np.vstack([((lsb3 >> i) & 1) for i in [0,1,2]]).T.flatten()[:N_BITS]
# pack bits into bytes and save
bit_bytes = np.packbits(bit_stream)
with open('trng_13M.bin', 'wb') as f:
    f.write(bit_bytes)
print(f"Generated {N_BITS} bits, saved to trng_13M.bin")

# Plot time‐series of raw samples
plt.figure()
plt.plot(raw, color='blue', linewidth=0.5)
plt.title("Raw Audio Samples")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

# Plot time‐series of bit_stream samples
plt.figure()
plt.plot(bit_stream, '.', markersize=1, color='green')
plt.title("Bit Stream Samples")
plt.xlabel("Sample Index")
plt.ylabel("Bit Value")

plt.show()

# Entropy calculation
cnt = Counter(lsb3)
prob = np.array([cnt[i]/len(lsb3) for i in range(8)])
H = -np.sum(prob * np.log2(prob + 1e-12))
print("Entropia 3 LSB:", H, "bit/B (max 3)")


