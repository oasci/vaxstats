# Rate-of-Change Thresholds

This document provides a comprehensive explanation of how **Rate-of-Change (ROC) Thresholds** are derived from [`SSSD`](./sssd.md), detailing their mathematical foundations, significance, and implementations.
Understanding these thresholds is essential for effectively identifying significant rates of temperature change, which are critical indicators in vaccine monitoring and safety assessments.

## Data Overview

The analysis begins with baseline temperature data, denoted as $B = [B_0, B_1, B_2, \ldots, B_N]$, where each $B_i$ represents a temperature measurement taken at a specific time interval, typically every 15 minutes.
The total number of baseline measurements is denoted by $N$, corresponding to the variable `BaseLength`.
This dataset serves as the foundation for assessing inherent variability in temperature readings before any interventions or treatments are applied.

## Computational Methodology

The **Sum of Squared Successive Differences (SSSD)** quantifies the total variability within the baseline temperature data by summing the squared differences between consecutive measurements. Mathematically, it is expressed as:

$$
\text{SSSD} = \sum_{i=2}^{N} (B_i - B_{i-1})^2
$$

This summation provides a scalar value that encapsulates the total variability present in the baseline temperature data by accounting for the squared changes between each pair of consecutive measurements.

To facilitate comparative analyses across datasets of varying sizes, the SSSD is normalized by dividing it by the degrees of freedom ($N - 1$), resulting in the **Mean Squared Successive Difference (MSSD)**:

$$
\text{MSSD} = \frac{\text{SSSD}}{N - 1}
$$

This normalization yields an average measure of squared changes per time point, enabling the establishment of standardized thresholds for identifying significant rates of temperature change.

## Thresholds

The establishment of these ROC Thresholds based on `SSSD` and `MSSD` ensures that only substantial deviations are considered significant. Minor fluctuations within the range defined by the ROC Upper and Lower Thresholds are treated as normal variability inherent in the baseline period. This methodological approach minimizes false positives—instances where normal variations are incorrectly identified as significant deviations—and ensures that only meaningful temperature changes trigger further investigation or intervention.

### Upper

The ROC upper threshold is defined as three times the square root of the Mean Squared Successive Difference (MSSD):

$$
\text{ROC Upper Threshold} = 3 \times \sqrt{\text{MSSD}}
$$

The choice of multiplying the square root of MSSD by three aligns with statistical principles aimed at capturing extreme deviations. Specifically, in a normally distributed dataset, approximately 99.7% of the data points lie within three standard deviations of the mean. By applying this multiplier, the ROC Thresholds are set to identify temperature measurements that represent unusually rapid changes, which are unlikely to occur by chance and may signify critical health events.

This threshold serves as a benchmark to identify significant increases in the rate of temperature change relative to baseline variability. When a temperature residual (the difference between an observed temperature and the expected baseline) exceeds the ROC Upper Threshold, it indicates a notable upward deviation that may signify a fever event. This threshold ensures that only substantial increases, unlikely to result from normal fluctuations, are flagged for further investigation or intervention.

### Lower

Conversely, the ROC lower is the negative counterpart of the ROC upper Threshold:

$$
\text{ROC Lower Threshold} = -3 \times \sqrt{\text{MSSD}}
$$

This threshold is used to detect significant decreases in the rate of temperature change, potentially indicating hypothermia. Temperature residuals falling below the ROC Lower Threshold are flagged as noteworthy deviations from the baseline, distinguishing them from normal minor temperature drops.
