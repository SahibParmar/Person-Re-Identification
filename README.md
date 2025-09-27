# Person Re-Identification (Re-ID)

This repository explores **Person Re-Identification (Re-ID)** using **CLIP-based models** and a **UNet-enhanced variant**.  
The goal is to match identities across multiple cameras and challenging conditions such as occlusions.  

We experiment with two approaches:
- **CLIP + fc** → Fine-tuned CLIP backbone with a fully connected layer.
- **UNet + CLIP + fc** → Adding UNet features to enhance spatial details before CLIP + fc.

---

## 🚀 Results

Below are the **Rank-1 Accuracy** and **mAP** results (including same-camera retrievals) across four benchmark datasets:  

### Rank-1 Accuracy
| Model        | Market1501 | CUHK03 | DukeMTMC | Occluded-Duke |
|--------------|------------|--------|----------|---------------|
| CLIP + fc    | **98.87%** | **95.36%** | **93.00%** | **96.79%** |
| UNet+CLIP+fc | 97.98%     | 94.50% | 92.10%   | 96.20%        |

### mAP
| Model        | Market1501 | CUHK03 | DukeMTMC | Occluded-Duke |
|--------------|------------|--------|----------|---------------|
| CLIP + fc    | **81.72%** | **79.71%** | **72.21%** | **50.13%** |
| UNet+CLIP+fc | 78.91%     | 73.02% | 70.68%   | 49.03%        |

➡️ **Observation:** CLIP+fc outperforms the UNet+CLIP+fc variant on all datasets, especially in mAP.  

## ⚙️ Setup & Installation

**Clone this repo**
   ```bash
   git clone https://github.com/SahibParmar/Person-Re-Identification.git
   cd Person-Re-Identification
```

**🔮 Future Work**

Explore cross-modal Re-ID (text-to-person retrieval).

Experiment with transformer-based backbones.

Improve occlusion robustness with part-aware modeling.
