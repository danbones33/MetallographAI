# Industry Standards for Grain Size Measurement

This directory contains reference information about industry standards used for grain size measurement in metallography.

## Key Standards

### ASTM E112
Standard Test Methods for Determining Average Grain Size

This is the most widely used standard for grain size determination. It describes several methods:
- Comparison procedure
- Planimetric (Jeffries) procedure
- Intercept procedures (linear, circular)
- Heyn procedure

The standard includes reference charts and calculation methods to determine the grain size number G:
```
G = 1 + 3.322 * log10(n)
```
where n is the number of grains per square inch at 100× magnification.

### ISO 643
Steels — Micrographic determination of the apparent grain size

This European standard is similar to ASTM E112 but has some differences in methodology and reporting. It is commonly used for steel products.

### ASTM E1382
Standard Test Methods for Determining Average Grain Size Using Semiautomatic and Automatic Image Analysis

This standard specifically addresses computer-assisted and automated methods for grain size determination, making it particularly relevant for this application.

## Grain Size Scales

### ASTM Grain Size Number
The ASTM grain size number (G) is related to the number of grains per unit area:

| G | Grains/mm² at 1× | Avg. Diameter (μm) |
|---|------------------|-------------------|
| 1 | 15.5 | 254 |
| 2 | 31 | 180 |
| 3 | 62 | 127 |
| 4 | 124 | 90 |
| 5 | 248 | 63.5 |
| 6 | 496 | 44.9 |
| 7 | 992 | 31.8 |
| 8 | 1980 | 22.5 |
| 9 | 3970 | 15.9 |
| 10 | 7940 | 11.2 |

## Implementation Guidelines

When implementing grain size analysis according to these standards:

1. **Calibration**: Ensure proper pixel-to-micron calibration based on microscope magnification
2. **Thresholding**: Document the thresholding or edge detection methods used
3. **Validation**: Compare automated results with manual measurements on reference samples
4. **Reporting**: Include the specific standard followed and all required measurements in reports

## References

Place relevant standard documents or reference materials in this directory for easy access. 