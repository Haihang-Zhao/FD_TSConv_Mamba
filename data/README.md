#### `deal_data.py` — Data Preprocessing Script

This script standardizes raw captured data into a clean format suitable for training and analysis. Its main functions include:

* Base Conversion: Convert fields from hexadecimal/binary to decimal/float representations;
* Normalization/Standardization: Apply Min-Max scaling to ensure consistent feature scales;
* Time/Window Processing (optional): Resample and align by timestamp, generate fixed-length sliding windows;
* Outlier and Missing Value Handling: Interpolate or remove invalid entries to maintain data continuity;

