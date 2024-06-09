# MolTextQA: A Curated Question-Answering Dataset and Benchmark
This is the official repository for **MolTextQA: A Curated Question-Answering Dataset and Benchmark for Molecular Structure-Text Relationship Learning**, currently in submission at Neurips datasets and benchmarking track.

## Contents
- [Accessing the dataset](#dataset-link)
- [Loading the Dataset](#loading-the-dataset)
- [Sample Data Point](#sample-data-point)
- [Dataset Categories](#dataset-categories)
- [Data Statistics](#data-statistics)
- [Citing the Dataset](#citing-the-dataset)

### Accessing the dataset
[Access the dataset here](https://huggingface.co/datasets/sl160/MolTextQA)

### Loading the Dataset
To load the dataset using the Hugging Face `datasets` library, you can use the following code:

```python
from datasets import load_dataset

dataset = load_dataset("sl160/MolTextQA")
```

### Sample Data Point
| Attribute     | Data                                                              |
|---------------|-------------------------------------------------------------------|
| SMILES sequence | CC(=O)C                                                           |
| Question      | What is the physical state of the molecule at room temperature?   |
| Options       | (a) Liquid (b) Solid (c) Gas                                      |
| Correct Option | (a) Liquid                                                        |
| Sentence      | The physical state of the molecule is liquid.                     |
| SMILES options | (a) CH4 (b) CC(=O)C (c) C(=O)([O-])[O-].[Ca+2]                    |
| Correct SMILE | (b) CC(=O)C                                                       |
| PubChem ID    | 180                                                               |
| Category      | Physical Properties                                               |

### Dataset Categories
- **Chemical Information:** covers the chemical structure, functional groups, and chemical properties.
- **Physical Properties:** addresses the properties such as solubility, physical state, and odor.
- **Biological Information:** contains the molecules' role in biological pathways, drug applications, and drug toxicity.
- **Source:** details the molecules' origin and manufacturing processes.
- **Application:** describes application areas such as perfumes, fertilizers, and insecticides.

### Data Statistics
| Splits    | Molecules | Total QAs | Physical Properties | Chemical Information | Biological Information | Source | Application |
|-----------|-----------|-----------|---------------------|----------------------|------------------------|--------|-------------|
| Pretrain  | 213336    | 429621    | 40221               | 187575               | 39473                  | 148867 | 13466       |
| Train     | 20000     | 55454     | 3741                | 28136                | 10528                  | 12124  | 925         |
| Valid     | 2500      | 5851      | 380                 | 2810                 | 910                    | 1659   | 92          |
| Test      | 5000      | 12092     | 749                 | 5947                 | 2003                   | 3206   | 187         |
| **Total** | 240836    | 503018    | 45091               | 224468               | 52914                  | 165856 | 14670       |

### Citing the Dataset
Please cite this dataset using the following BibTeX entry:

```bibtex
@misc{siddhartha_laghuvarapu_2024,
	author = {Siddhartha Laghuvarapu},
	title = {MolTextQA (Revision 7ec025f)},
	year = 2024,
	url = {https://huggingface.co/datasets/sl160/MolTextQA},
	doi = {10.57967/hf/2443},
	publisher = {Hugging Face}
}
