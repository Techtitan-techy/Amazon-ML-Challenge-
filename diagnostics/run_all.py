import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step01_profile_and_uniqueness import run_step01
from step02_normalization_experiments import run_step02
from step03_blocking_evaluation import run_step03
from step04_pair_analysis import run_step04
from step05_modeling_and_decisions import run_step05
from step06_architecture_recommendation import run_step06

def main():
    print("#" * 70)
    print("AMAZON ML CHALLENGE 2026 - COMPLETE DIAGNOSTIC PIPELINE")
    print("#" * 70)
    
    t_start = time.time()
    
    steps = [
        ("Step 1: Data Profile, Uniqueness & Distributions", run_step01),
        ("Step 2: Normalization Experiments", run_step02),
        ("Step 3: Blocking Recall, Cost & Pareto", run_step03),
        ("Step 4: True Pairs, Hard Negatives & Feature Separability", run_step04),
        ("Step 5: Model Benchmark, Thresholds & Decisions", run_step05),
        ("Step 6: Architecture Recommendation Synthesis", run_step06),
    ]
    
    for idx, (name, fn) in enumerate(steps, 1):
        step_t0 = time.time()
        print(f"\n>>> Starting [{idx}/{len(steps)}] {name}...")
        try:
            fn()
            print(f">>> Completed [{idx}/{len(steps)}] {name} in {time.time()-step_t0:.2f}s")
        except Exception as e:
            print(f"!!! Error in [{idx}/{len(steps)}] {name}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    print("\n" + "#" * 70)
    print(f"ALL DIAGNOSTIC STEPS FINISHED in {time.time()-t_start:.2f}s")
    print("All 18 reports and figures are saved in the `reports/` directory.")
    print("#" * 70)

if __name__ == "__main__":
    main()
