1. Purpose
benchmark all TensorRT engines and measure FPS + latency using benchmark.py script with interactive run count input

2. Prerequisites
    Engines built
    the project's directory, referenced as [project]

3. Expected Output
a series of results
A summary table like:
=== Summary ===
MultiTask    |  XX.XX FPS |  XX.XX ms
det          |  XX.XX FPS |  XX.XX ms
seg          |  XX.XX FPS |  XX.XX ms
depth        |  XX.XX FPS |  XX.XX ms

4. Instructions

4.1 
open the command prompt

4.2
enter the command:
    python3 benchmark.py

when prompted enter the number of test runs as a number, e.g.:
100

5. Verification
    FPS values should be stable
    No crashes during warm-up or timed runs

6. Common Errors & Fixes
Error: ValueError from input
Fix: the script defaults to 100 runs

Error: Engine fails to load
Fix: rebuild that engine and try again

7. Next Steps
04_node_info.MD
05_troubleshooting.md

