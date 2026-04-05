import time
import numpy as np
from trtInferUtils import TRTInference

# Engines to benchmark
MODELS = {
    "multiTask": "multiTaskModel.engine",
    "det": "singleDet.engine",
    "seg": "singleSeg.engine",
    "depth": "singleDepth.engine"
}

def benchmarkEngine(name, enginePath, warmup = 10, runs = 100):
    print(f"\n=== Benchmarking {name} ===")
    trtModel = TRTInference(enginePath)

    # Dummy input
    img = np.random.randn(1, 3, 480, 640).astype(np.float32)

    # Warm-up
    for _ in range(warmup):
        trtModel.infer(img)

    img = []
    for i in runs:
        img[i] = np.random.randn(1, 3, 480, 640).astype(np.float32)
    
    # Timed runs
    timedRunsInd = []
    indivRunNext = time.time()
    indivRunprev = time.time()
    start = time.time()
    for i in range(runs):
        trtModel.infer(img[i])
        indivRunNext = time.time()
        timedRunsInd[i] = indivRunNext - indivRunprev
        indivRunprev = indivRunNext
    end = time.time()
    totalRuns = 0.0
    for x in timedRunsInd:
        totalRuns = totalRuns + x 
    totalTime = end - start
    avgTotalTime = totalRuns / runs
    totalfps = 1.0
    avgTime = totalTime / runs
    fps = 1.0 / avgTime
    print(f"Avg inference time: {avgTime*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Avg actual time: {avgTotalTime*1000:.2f} ms")
    print(f"avg actual fps: {totalfps:.2f}")
    return fps, avgTime, totalfps, avgTotalTime, timedRunsInd


def main():
    results = {}

    for name, engine in MODELS.items():
        fps, avg, _, _, _ = benchmarkEngine(name, engine)
        results[name] = (fps, avg)

    print("\n=== Summary ===")
    for name, (fps, avg) in results.items():
        print(f"{name:10s} | {fps:6.2f} FPS | {avg*1000:6.2f} ms")
    
if __name__ == "__main__":
    main()