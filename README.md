# Assignment 5: ML-Based Taxiway Marking Detection Using ALINA

## 1. Project Overview

This project extends the ALINA aircraft taxiway line detection system from Assignment 4. The goal of Assignment 5 is to explore machine learning techniques for identifying yellow and white taxiway markings instead of relying only on manually selected color thresholds.

Five clustering methods were implemented and compared:

1. K-Means
2. Gaussian Mixture Model (GMM)
3. DBSCAN
4. Fuzzy C-Means
5. Mean Shift

Each method processes airport images and produces separate binary masks for yellow and white markings.

**Note:** The current implementation uses machine learning for color clustering, but still uses broad, manually defined HSV conditions to identify yellow and white clusters. Therefore, it is a clustering-based alternative to direct manual thresholding, rather than a completely automatic threshold-learning system.

## 2. ML Pipeline

The implemented pipeline follows these steps:

1. **Input images:** Load airport frames from the selected video dataset.
2. **ROI selection:** Load the previously saved GMM region of interest (ROI) from Assignment 4.
3. **Image resizing:** Resize images to a maximum dimension of 480 pixels to reduce processing time.
4. **Feature extraction:** Convert images from BGR to LAB color space and normalize pixel values.
5. **ML clustering:** Apply one of the five clustering algorithms to group pixels with similar colors.
6. **Color identification:** Use broad HSV color conditions on the average color of each cluster to identify yellow and white candidates.
7. **Mask filtering:** Remove small connected components and regions larger than 10% of the ROI area.
8. **Output generation:** Save separate yellow and white masks, visualization previews, ROI masks, and processing-time measurements.

The same saved GMM ROI is used for all five methods to provide a consistent region for comparison. In the current implementation, clustering is performed on the resized image, while the ROI is applied during cluster selection and mask filtering.

## 3. Project Structure

```text
ALINA_New/
├── alina/
│   ├── color_detection.py
│   ├── color.py
│   ├── config.py
│   └── roi.py
├── experiments/
│   └── Assignment_5/
│       └── run_color_method.py
├── data/
│   └── Raw_Data/
│       ├── vidd_1/
│       ├── vidd_2/
│       └── vidd_3/
├── outputs/
│   ├── gmm/
│   │   ├── vidd_1_roi.txt
│   │   ├── vidd_2_roi.txt
│   │   └── vidd_3_roi.txt
│   └── assignment_5/
│       ├── kmeans/
│       ├── gmm/
│       ├── dbscan/
│       ├── fuzzy_cmeans/
│       └── meanshift/
└── README.md
```

The `outputs/gmm/` directory contains the ROI files generated in Assignment 4.

## 4. Installation

The experiments were developed using Python 3.12.

From the project root, create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
pip install numpy==1.26.4 opencv-python==4.10.0.84 scikit-learn==1.5.2
```

The repository must also contain the ALINA modules and image dataset in the directory structure shown above.

## 5. Running the Experiments

All commands should be executed from the `ALINA_New` project root.

**K-Means**

```powershell
python experiments/Assignment_5/run_color_method.py --method kmeans --video vidd_1
```

**Gaussian Mixture Model (GMM)**

```powershell
python experiments/Assignment_5/run_color_method.py --method gmm --video vidd_1
```

**DBSCAN**

```powershell
python experiments/Assignment_5/run_color_method.py --method dbscan --video vidd_1
```

**Fuzzy C-Means**

```powershell
python experiments/Assignment_5/run_color_method.py --method fuzzy_cmeans --video vidd_1
```

**Mean Shift**

```powershell
python experiments/Assignment_5/run_color_method.py --method meanshift --video vidd_1
```

By default, the program loads:

```text
outputs/gmm/vidd_1_roi.txt
```

To specify a different saved ROI:

```powershell
python experiments/Assignment_5/run_color_method.py --method gmm --video vidd_1 --roi-file outputs/gmm/vidd_1_roi.txt
```

Replace `vidd_1` with `vidd_2` or `vidd_3` to run experiments on the other datasets.

## 6. Reproducibility

A fixed random seed is used in `alina/color_detection.py`:

```python
SEED = 42
```

The same seed controls random pixel sampling and the random initialization of supported clustering methods.

The main experimental settings are:

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Maximum image dimension | 480 pixels |
| Maximum sampled pixels | 6000 |
| K-Means clusters | 12 |
| GMM components | 12 |
| DBSCAN `eps` | 0.055 |
| DBSCAN `min_samples` | 12 |
| Fuzzy C-Means centers | 12 |
| Mean Shift bandwidth | 0.12 |

The experiments use the same input frames and saved GMM ROI for comparison. Results may still vary slightly across different hardware or library environments.

## 7. Experimental Results

The following results were obtained from 50 frames of `vidd_1`.

| ML Method | Frames with Yellow Pixels | Frames with White Pixels | Average Processing Time |
|---|---|---|---|
| K-Means | 0/50 | 50/50 | 206.67 ms/frame |
| GMM | 46/50 | 49/50 | 256.53 ms/frame |
| DBSCAN | 6/50 | 0/50 | 672.74 ms/frame |
| Fuzzy C-Means | 0/50 | 50/50 | 146.35 ms/frame |
| Mean Shift | 0/50 | 0/50 | 163.20 ms/frame |

These results measure the number of frames containing at least one detected pixel for each color. They do not represent detection accuracy, precision, recall, or F1-score.

### Results and Discussion

GMM produced nonempty yellow masks in 46 out of 50 frames and nonempty white masks in 49 out of 50 frames. Based on these results and visual inspection, GMM was selected as the most promising method for further evaluation.

K-Means and Fuzzy C-Means produced white masks in all 50 frames but did not identify yellow pixels. DBSCAN detected yellow pixels in only six frames and was the slowest method. Mean Shift did not produce yellow or white detections with the tested settings.

Fuzzy C-Means had the lowest average processing time, but faster processing did not necessarily lead to better detection.

The results suggest that clustering performance depends strongly on the algorithm, its parameters, and the method used to identify yellow and white clusters.

## 8. Generated Outputs

Each experiment saves results in:

```text
outputs/assignment_5/<method>/<video>/
```

For example:

```text
outputs/assignment_5/gmm/vidd_1/
├── yellow_masks/
├── white_masks/
├── yellow_previews/
├── white_previews/
├── roi_masks/
└── results.csv
```

The yellow and white masks are binary images in which white pixels represent detected candidates.

The preview images highlight detected pixels on the original frame.

The `results.csv` file contains:

- Frame filename
- ML method
- Number of yellow pixels
- Number of white pixels
- Processing time in milliseconds

These files can be used to compare the five methods and create performance graphs.

## 9. Limitations

The current implementation has several limitations.

First, clustering does not directly determine which pixels belong to taxiway markings. Broad HSV conditions are still used to identify yellow and white clusters.

Second, the color of an entire cluster is represented by its average color. If a cluster contains both pavement and markings, the average may not accurately represent either.

Third, the algorithm may detect bright aircraft surfaces or other objects instead of white taxiway lines.

Fourth, the current experiments do not include independently annotated ground-truth masks. Therefore, precision, recall, F1-score, and Intersection over Union (IoU) have not yet been measured.

Finally, the clustering algorithms were tested with selected parameter values. Additional tuning and evaluation may improve their results.

## 10. Future Improvements

Possible improvements include clustering only pixels inside the ROI, improving cluster identification, testing different algorithm parameters, and learning color boundaries directly from selected clusters.

A more complete evaluation should compare the detected masks with manually annotated taxiway markings and measure precision, recall, F1-score, and IoU.

The best-performing method should also be integrated into the complete ALINA line-detection pipeline and demonstrated using output videos.

## 11. Conclusion

Five machine learning clustering techniques were implemented and evaluated for yellow and white taxiway marking detection.

Among the tested methods, GMM gave the most promising results for detecting both colors, while the other methods showed limitations under the current settings.

The experiments demonstrate how machine learning clustering can be used as part of ALINA's color-detection stage. Further work is needed to learn color thresholds more automatically and validate the complete line-detection performance.