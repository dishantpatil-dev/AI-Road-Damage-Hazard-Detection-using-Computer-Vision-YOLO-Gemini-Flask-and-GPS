# CivicScan performance benchmark

The benchmark measures the end-to-end detector call for sampled video frames.
Each sampled frame is sent through all three configured Roboflow workflows.

Metrics reported:
- frames processed
- elapsed time
- average per-frame latency
- p95 per-frame latency
- end-to-end FPS
- prediction count
- average prediction confidence
- maximum prediction confidence

## Run

Configure the normal Roboflow environment variables first, then:

```bash
python benchmarks/run_benchmark.py --video path/to/representative-road-video.mp4 --max-frames 40 --sample-every 2
```

Write machine-readable results:

```bash
python benchmarks/run_benchmark.py \
  --video path/to/representative-road-video.mp4 \
  --max-frames 40 \
  --sample-every 2 \
  --output benchmark-results.json
```

For a fair comparison, use the same representative video, frame sampling interval,
maximum frame count, detector configuration and network environment between runs.

No video is committed to the repository because benchmark footage can contain
third-party/copyrighted material and because the detector calls consume the
configured inference service.
