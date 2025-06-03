# Knowledge Base for Grain Size Analyzer

This directory contains resources and training materials for the grain size analysis models.

## Structure

- `/datasets`: Example annotated datasets for training
- `/papers`: Relevant academic papers and resources
- `/tutorials`: Guides on model training and annotation
- `/standards`: Industry standards for grain size measurement (e.g., ASTM E112)

## Adding New Training Data

1. Place new microscopic images in a subdirectory under `/datasets`
2. Create annotation files for each image (JSON or YOLO format)
3. Document the dataset characteristics in a README file

## Annotation Format

For grain boundaries, annotations should be stored in one of these formats:

### JSON Format (Recommended)
```json
{
  "image": "sample_001.jpg",
  "width": 1024,
  "height": 768,
  "grains": [
    {
      "id": 1,
      "boundary": [[x1, y1], [x2, y2], ..., [xn, yn]],
      "size": 15.6,
      "category": "alpha"
    },
    ...
  ],
  "metadata": {
    "magnification": "500x",
    "etchant": "Nital 2%",
    "material": "Steel 1045"
  }
}
```

### YOLO Format
```
# class x_center y_center width height
0 0.716797 0.395833 0.216406 0.147222
0 0.287109 0.563368 0.255469 0.168056
...
```

## Training Knowledge

This section can be expanded with specific information about:

1. Grain size measurement methodologies
2. Different material types and their grain characteristics
3. Best practices for preparing metallographic samples
4. Reference charts for visual comparison

## Reference Standards

Common standards for grain size measurement:

- ASTM E112: Standard Test Methods for Determining Average Grain Size
- ISO 643: Steels - Micrographic determination of the apparent grain size
- ASTM E1382: Standard Test Methods for Determining Average Grain Size Using Semiautomatic and Automatic Image Analysis 