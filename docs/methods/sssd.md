# Sum of Squared Successive Differences (SSSD)

In the realm of vaccine statistics, accurately assessing temperature variability during the baseline period is essential for ensuring the reliability of subsequent analyses.
A critical metric utilized for this purpose is the **Sum of Squared Successive Differences (SSSD)**, referred to as `SSSD` in this documentation.
This document elucidates the mathematical underpinnings and computational procedures employed to calculate `SSSD` from baseline temperature data, thereby offering a comprehensive understanding of its role and significance in temperature data analysis.

!!! warning

    Previous MATLAB and R code called this metric "Residual Sum of Squares", which is incorrect.

## Data Overview

The analysis begins with baseline temperature data, denoted as $B = [B_0, B_1, B_2, \ldots, B_N]$, where each $B_i$ represents a temperature measurement taken at some time interval (usually 15 minutes).
The total number of baseline measurements is denoted by $N$, corresponding to the variable `BaseLength`.
This dataset serves as the foundation for assessing the inherent variability in temperature readings before any interventions or treatments are applied.

## Computational Methodology

The computation of `SSSD` involves a systematic process that begins with calculating the first differences between consecutive temperature measurements.
For each time point $i$ ranging from 2 to $N$, the difference $\Delta B_i = B_i - B_{i-1}$ is computed.
This difference quantifies the change in temperature from one measurement to the next, providing insight into the day-to-day or period-to-period fluctuations in temperature.

However, real-world data often contain missing values, represented as `NaN` (Not a Number).
To ensure the robustness of the `SSSD` calculation, the methodology accounts for these missing values by setting the corresponding difference $\Delta B_i$ to zero whenever either $B_i$ or $B_{i-1}$ is `NaN`.
This approach effectively excludes incomplete data points from influencing the `SSSD`, thereby maintaining the integrity of the variability assessment.

Once the first differences are established, each valid difference is squared to eliminate the effect of directionality, ensuring that both positive and negative changes contribute positively to the total sum.
This squaring operation results in a series of squared differences, which are then aggregated to compute the **Sum of Squared Successive Differences (SSSD)**.
Mathematically, this is expressed as:

$$
\text{SSSD} = \sum_{i=2}^{N} (B_i - B_{i-1})^2.
$$

This summation provides a scalar value that encapsulates the total variability present in the baseline temperature data by summing the squared changes between consecutive measurements.

To normalize this sum and facilitate comparative analyses, especially when dealing with datasets of varying sizes, the code calculates the mean squared difference by dividing `SSSD` by the degrees of freedom, which is $N - 1$.
This normalization yields `SSSDVal`, offering an average measure of squared changes per time point.
This metric is particularly useful for setting thresholds or assessing the variability relative to the size of the baseline dataset.

## Implementations

### MATLAB

```matlab
% Initialize SumSquaredSuccessiveDiffs vector
SumSquaredSuccessiveDiffs = zeros(BaseLength,1);

% Compute squared differences between consecutive baseline temperatures
for i = 2:BaseLength
    if isnan(BaseLine(i)) || isnan(BaseLine(i-1))
        SumSquaredSuccessiveDiffs(i) = 0;  % Exclude NaN values
    else
        SumSquaredSuccessiveDiffs(i) = (BaseLine(i) - BaseLine(i-1))^2;  % Squared difference
    end
end

% Compute Sum of Squared Successive Differences (SSSD)
SSSD = sum(SumSquaredSuccessiveDiffs(1:BaseLength));

% Compute Mean Squared Difference
SSSDVal = SSSD / (BaseLength - 1);
```

### R

```R
# Calculate Sum of Squared Successive Differences (SSSD)
SumSquaredSuccessiveDiffs <- data.frame()
for (x in 2:length(Baseline)) {
    S1 <- Baseline[x]
    S2 <- Baseline[x-1]
    S3 <- ((S1 - S2)^2)
    if (is.na(S3)) {
        SumSquaredSuccessiveDiffs <- rbind(SumSquaredSuccessiveDiffs, 0)
    } else {
        SumSquaredSuccessiveDiffs <- rbind(SumSquaredSuccessiveDiffs, S3)
    }
}
colnames(SumSquaredSuccessiveDiffs) <- ("SumSquaredSuccessiveDiffs")

# Calculate the Sum of Squared Successive Differences (SSSD)
SSSD <- sum(SumSquaredSuccessiveDiffs$SumSquaredSuccessiveDiffs)

# Calculate Mean Squared Difference
SSSDVal <- SSSD / (length(Baseline) - 1)
```
