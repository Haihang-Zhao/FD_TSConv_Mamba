#### `deal_data.py` — Data Preprocessing Script

This script standardizes raw captured data into a clean format suitable for training and analysis. Its main functions include:

* Base Conversion: Convert fields from hexadecimal/binary to decimal/float representations;
* Normalization/Standardization: Apply Min-Max scaling to ensure consistent feature scales;
* Time/Window Processing (optional): Resample and align by timestamp, generate fixed-length sliding windows;
* Outlier and Missing Value Handling: Interpolate or remove invalid entries to maintain data continuity;



#### `label-determination.ipynb` — AVTPDU Parsing & Conventional Replay Attack Generation

Parses raw in-vehicle Ethernet **AVTPDU** frames and synthesizes **replay** attack samples, producing labeled datasets ready for model training/evaluation.

**Core functions**

* Protocol parsing:** Parse IEEE 1722/AVTP fields (`stream_id`, `subtype`, `sv`, `sequence_num`, `presentation_time`, `payload`), align timestamps, and perform basic validation.
* Session filtering & preprocessing:** Select target flows by `stream_id`/ports, deduplicate and denoise records, and organize sequences by time or fixed windows.
* Replay attack synthesis:** Extract segments from benign traffic and re-inject them in original order and inter-arrival timing (conventional replay). Configurable window length, repeat count, and delay strategy (fixed/jittered).
* Labeling & export:** Assign labels (e.g., `0 = normal`, `1 = replay`) and export standardized datasets (CSV/NPZ) with summary statistics.
* (Optional) Visualization:** Quick time-series and distribution plots for data-quality checks.

**Purpose:** Transform raw AVTPDU captures into structured, reproducible replay-attack datasets to support intrusion detection and anomaly-detection research.

