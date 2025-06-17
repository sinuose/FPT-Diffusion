# 📈 Gaussian Single Particle Tracking (SPT) Analysis

This repository contains a modular and robust pipeline for **analyzing widefield single-particle tracking (SPT) data** using 2D Gaussian fitting and background signal estimation. It is designed for use with real or synthetic video frames and produces refined localization data suitable for downstream analysis.

---

## 🚀 Features

- 🔍 **Subpixel Particle Localization**  
  Fit 2D Gaussians to bright spots in noisy microscopy frames for high-precision tracking.
  
- 🧠 **Parameter Extraction**  
  Outputs key Gaussian fit parameters including centroid coordinates, widths, orientation, amplitude, and background offset.
- 📉 **Background Estimation Module**  
  Statistically estimates frame-by-frame background signal by masking regions around detected particles.

---

## 🧬 Pipeline Overview

### 1. **Gaussian Fitting with `sptObject`**

Once loaded with a video (3D NumPy array), the `sptObject` performs:

```
spt = sptObject(video)
spt.StandardSPT()
```

### 2. **Data Extraction**
```
numbers = spt.GetSptResults()
signal = spt.GetSigResults()
```
### 3. **📚Structure**

├── analysis_main.py         # Entry point for analysis
├── FPTdiffusion/
│   └── sptObject.py         # Core localization logic
├── README.md

## 4. Citations

The group at TrackPy is really good. Use their stuff first before trying this one. 
Trackpy (https://github.com/soft-matter/trackpy)




