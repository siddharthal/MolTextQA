# MolTextQA: A Curated Question-Answering Dataset and Benchmark
This is the official repository for **MolTextQA: A Curated Question-Answering Dataset and Benchmark for Molecular Structure-Text Relationship Learning**, currently under review.

## Contents
- [Accessing the dataset](#accessing-the-dataset)
- [Loading the Dataset](#loading-the-dataset)
- [Sample Data Point](#sample-data-point)
- [Dataset Categories](#dataset-categories)
- [Data Statistics](#data-statistics)
- [Benchmark](#benchmark)
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


## Benchmark

### Zero-shot
The Molecule QA task requires selecting the correct option from a set, given a SMILES string and a related question. The Molecule Retrieval task involves choosing the correct SMILES from candidates, based on a molecular property description. All the reported numbers are %correct answers (accuracy). 

| Model        | Entire dataset | Physical Properties | Chemical Info | Biological Info | Sources | Uses   |
|--------------|----------------|---------------------|---------------|-----------------|---------|--------|
| **Molecule QA**     |                |                     |               |                 |         |        |
| SciBERT      | 21.26          | 20.16               | 21.52         | 19.02           | 22.36   | 22.46  |
| KV-PLM       | 29.86          | 27.90               | 32.03         | 26.61           | 28.51   | 26.74  |
| MoleculeSTM  | 44.68          | 30.17               | 48.02         | 30.90           | 51.28   | 31.02  |
| MoMu         | 44.93          | 28.84               | 49.60         | 30.05           | 50.31   | 27.81  |
| gpt3.5       | 40.30          | 44.73               | 38.54         | 43.29           | 40.24   | 47.59  |
| llama3-8b    | 16.16          | 20.03               | 16.08         | 17.27           | 14.07   | 27.27  |
| llama3-70b   | 58.24          | 63.28               | 57.07         | 56.91           | 59.58   | 66.31  |
| llama2-7b    | 24.57          | 22.70               | 25.31         | 24.86           | 23.46   | 24.60  |
| llama2-70b   | 28.79          | 30.57               | 27.11         | 36.15           | 26.39   | 37.43  |
| Random       | 20.69          | 20.55               | 22.78         | 21.11           | 20.18   | 19.49  |
| **Molecule Retrieval** |                |                     |               |                 |         |        |
| SciBERT      | 21.22          | 21.36               | 21.93         | 21.12           | 19.84   | 22.99  |
| KV-PLM       | 48.53          | 47.80               | 60.01         | 43.09           | 30.47   | 54.01  |
| MoleculeSTM  | 67.38          | 49.13               | 76.98         | 53.02           | 63.57   | 54.55  |
| MoMu         | 66.02          | 45.79               | 76.51         | 51.12           | 61.51   | 50.27  |
| gpt3.5       | 38.31          | 39.92               | 47.02         | 31.10           | 26.36   | 36.90  |
| llama3-8b    | 20.96          | 22.16               | 21.93         | 20.12           | 19.68   | 16.04  |
| llama3-70b   | 52.72          | 41.26               | 70.32         | 38.14           | 32.63   | 39.57  |
| llama2-7b    | 18.58          | 19.36               | 19.24         | 17.57           | 17.97   | 16.04  |
| llama2-70b   | 20.35          | 16.56               | 21.93         | 18.22           | 19.93   | 15.51  |
| Random       | 20.28          | 20.43               | 21.12         | 19.84           | 20.58   | 19.17  |



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
