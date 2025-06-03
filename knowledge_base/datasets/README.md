# Training Datasets

Place your training datasets in this directory. Each dataset should be organized as follows:

```
/dataset_name/
  ├── images/
  │    ├── sample001.jpg
  │    ├── sample002.jpg
  │    └── ...
  │
  ├── annotations/
  │    ├── sample001.json
  │    ├── sample002.json
  │    └── ...
  │
  └── metadata.json
```

## Metadata Format

The `metadata.json` file should contain information about the dataset:

```json
{
  "name": "steel_samples_2023",
  "description": "Carbon steel grain structure samples at 500x magnification",
  "source": "Metallurgy Lab",
  "image_count": 120,
  "image_size": [1024, 768],
  "material_type": "Carbon Steel",
  "magnification": "500x",
  "etchant_used": "Nital 2%",
  "grain_types": ["ferrite", "pearlite"],
  "label_format": "json_boundaries",
  "date_collected": "2023-05-15"
}
```

## Example Datasets

A few small example datasets are included to help with initial setup and testing:

1. **steel_samples_basic**: Simple carbon steel samples with clear grain boundaries
2. **aluminum_samples**: Aluminum alloy samples with different grain structures
3. **test_dataset**: Very small dataset for quick testing of application features 