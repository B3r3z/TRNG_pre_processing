import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ccml # Zaimportowano moduł ccml

raw = np.fromfile('raw_audio_u8_44k.u8', dtype=np.uint8) # Read raw audio samples, change the file name as needed
offset = 10000
N_BITS = 13000000
samples_needed = int(np.ceil(N_BITS / 3))
raw = raw[offset:offset + samples_needed]

lsb3 = raw & 0b00000111
# Save raw 3 LSB values as uint8 file
lsb3.tofile('source.u8')
# Extract bits (LSB first per sample)
bit_stream = np.vstack([((lsb3 >> i) & 1) for i in [0,1,2]]).T.flatten()[:N_BITS]
# Pack bits into bytes and save
bit_bytes = np.packbits(bit_stream)
with open('source.bin', 'wb') as f:
    f.write(bit_bytes)
print(f"Generated {N_BITS} bits, saved to source.bin")


# Histogram for raw audio samples
plt.figure()
plt.hist(raw, bins=256, color='blue', alpha=0.7)
plt.title("Histogram of Raw Audio Samples")
plt.xlabel("Sample Value")
plt.ylabel("Frequency")
plt.yscale('log')
plt.savefig("raw_audio_histogram.png") # Zapis wykresu do pliku
plt.close() # Zamknięcie figury

# Probability distribution for raw audio samples
plt.figure(figsize=(10, 6)) # Adding figure size for better readability
counts_raw = Counter(raw)
# Make sure we have probabilities for all possible values from 0 to 255
probabilities_raw = np.array([counts_raw[i]/len(raw) if len(raw) > 0 else 0 for i in range(256)])
plt.bar(range(256), probabilities_raw, color='darkslateblue', width=1.0)
plt.title("Empirical Probability Distribution of Raw Audio Samples")
plt.xlabel("Sample Value (x)")
plt.ylabel("Occurrence Frequency (p_i)")
plt.xlim([-0.5, 255.5]) # Better X-axis fit
# plt.ylim([0, max(probabilities_raw) * 1.1]) # Optional: adjust Y-axis
plt.savefig("raw_audio_distribution.png") # Zapis wykresu do pliku
plt.close() 

## Histogram for 3 LSB values
#plt.figure()
#plt.hist(lsb3, bins=8, range=(-0.5, 7.5), color='purple', alpha=0.7, rwidth=0.8) # 8 bins for values 0-7
#plt.title("Histogram of 3 LSB Values")
#plt.xlabel("3 LSB Value (0-7)")
#plt.ylabel("Frequency")
#plt.xticks(range(8))
#plt.yscale('log')
#plt.savefig("lsb3_histogram.png") # Zapis wykresu do pliku
#plt.close() # Zamknięcie figury

# Entropy calculation
cnt = Counter(lsb3)
# Calculate probabilities for all 8 possible symbols (0-7)
probabilities = np.array([cnt[i]/len(lsb3) for i in range(8)])

# Calculate entropy, considering only non-zero probabilities,
# because the term p*log2(p) is 0 when p=0.
entropy_terms = []
for p_i in probabilities:
    if p_i > 0:
        entropy_terms.append(p_i * np.log2(p_i))

# If lsb3 list is empty or all probabilities are 0
if not entropy_terms:
    H = 0.0
else:
    H = -np.sum(entropy_terms)

print("Entropy of 3 LSB:", H, "bits per symbol")


ccml.run_ccml(filename="source.bin", 
               output_filename="post.bin", 
               N_target_bits=N_BITS, 
               plot_filename="ccml_post_bin_distribution.png")

print("Pre-processing and CCML processing complete. Plots saved to files.")


