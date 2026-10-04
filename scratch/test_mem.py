import sys, gc

# Simulate target_best dictionary with 5,700,000 entries
print("Allocating 5.7M entries in dictionary...")
d = {}
for i in range(5700000):
    d[f"S2-{i}"] = (f"S1-{i % 1700000}", 0.85)

print(f"Dictionary length: {len(d):,}")
print("Checking memory footprint...")
import psutil, os
process = psutil.Process(os.getpid())
print(f"RAM Used: {process.memory_info().rss / (1024*1024):.2f} MB")
