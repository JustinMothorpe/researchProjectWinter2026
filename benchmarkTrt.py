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
    _, _, _, inputShape = trtModel.inputs[0]
    # Dummy input
    img = np.random.randn(*inputShape).astype(np.float32)
    #print(f"[BENCH] Generated img.shape: {img.shape}, size: {img.size}")

    # Warm-up
    for _ in range(warmup):
        trtModel.infer(img)

    img = []
    for _ in range(runs):
        newImg = np.random.randn(
                *inputShape
            ).astype(
                np.float32
            )
        img.append(
            newImg
        )
        #print(f"[BENCH] Generated img.shape: {newImg.shape}, size: {newImg.size}")

    
    # Timed runs
    timedRunsInd = []
    prev = time.time()
    start = prev

    for i in range(runs):
        #print(f"[BENCH] Using input shape from engine: {img[i].shape}")
        #print(f"[BENCH] Total elements: {img[i].shape}")

        trtModel.infer(img[i])
        now = time.time()
        timedRunsInd.append((now,prev))
        prev = now

    end = time.time()
    
    totalRuns = 0.0
    for x in timedRunsInd:
        first, last = x
        totalRuns = totalRuns + first - last
        
    totalTime = end - start
    avgTotalTime = totalRuns / runs
    totalfps = 1.0 / avgTotalTime
    avgTime = totalTime / runs
    fps = 1.0 / avgTime

    print(f"Avg inference time: {avgTime*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Avg actual time: {avgTotalTime*1000:.2f} ms")
    print(f"avg actual fps: {totalfps:.2f}")
    print(f"individual runs:")
    print(f"from start time: {start*1000:.2f} ms")
    
    iterations = 1
    
    for x in timedRunsInd:
        first, last = x
        print(
            f"run {iterations} | start time: {(first-start)*1000:.2f}"
            f" ms | finish: {(last-start)*1000:.2f}"
        )
        
        iterations= iterations+1
    
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