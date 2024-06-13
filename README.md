# MolTextQA: A Curated Question-Answering Dataset and Benchmark
This is the official repository for **MolTextQA: A Curated Question-Answering Dataset and Benchmark for Molecular Structure-Text Relationship Learning**, currently under review.

## Contents
- [Accessing the dataset](#accessing-the-dataset)
- [Loading the Dataset](#loading-the-dataset)
- [Dataset Categories](#dataset-categories)
- [Data Statistics](#data-statistics)
- [Data Structure](#data-structure)
- [Intended Uses](#intended-uses)
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

### Data structure

Each data point in the dataset contains the following fields:

- **CID**: The PubChem Identifier of the molecule.
- **QID**: The identifier of the question within a CID.
- **Category**: The category of the data point, following this convention:
  1. Physical properties
  2. Chemical information
  3. Biological uses
  4. Sources
  5. General applications
- **Sentence**: A sentence summary of the question and answer.
- **Question**: The actual question asked.
- **Options**: A set of options for the answer, of which one is correct.
- **Correct_option**: The index (1-based) of the correct option.
- **Retrieval_options**: A set of PubChem IDs used for molecule retrieval from the sentence task.
- **Retrieval_correct**: The correct option in the retrieval task.

### Intended Uses 

The dataset is primarily intended to be used for molecule-text relationship learning. The task of molecule-text learning has been gaining increasing attention in recent research. However, the current datasets and developed models do not enable structured inference, and evaluation is not precise. The MolTextQA dataset addresses these challenges by offering a question-answering format with multiple-choice answers. Questions are based on a small molecule input, with answers provided in textual sentence or multiple-choice format. The dataset is intended for applications in fields such as drug discovery, retrosynthesis, and the discovery of materials like fertilizers, pesticides, and perfumes.

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

### Finetuning Performance

Accuracy of different models in the finetuning setting, in both Molecule QA and Molecule Retrieval tasks.

| Model            | Entire dataset | Physical Properties | Chemical Info | Biological Info | Sources | Uses   |
|------------------|----------------|---------------------|---------------|-----------------|---------|--------|
| **Molecule QA**          |                |                     |               |                 |         |        |
| MoleculeSTM      | 65.14          | 68.62               | 61.86         | 65.35           | 69.93   | 71.12  |
| MoMu             | 65.08          | 70.76               | 60.69         | 66.65           | 70.56   | 71.66  |
| Llama3-8b        | 60.41          | 64.35               | 58.67         | 64.10           | 60.73   | 55.08  |
| Llama2-7b        | 41.84          | 42.06               | 43.97         | 41.64           | 38.21   | 37.43  |
| Galactica-125m   | 43.97          | 43.39               | 43.13         | 43.58           | 46.29   | 37.43  |
| Galactica-1.3b   | 60.98          | 62.62               | 58.60         | 62.41           | 64.85   | 48.66  |
| Galactica-6.7b   | 69.01          | 70.36               | 65.73         | 72.99           | 72.52   | 65.24  |
| Molt5-large      | 34.15          | 30.57               | 34.27         | 38.34           | 32.56   | 26.74  |
| Molt5-large-s2c  | 34.69          | 47.26               | 31.16         | 37.89           | 36.49   | 31.55  |
| Random           | 20.69          | 20.55               | 22.78         | 21.11           | 20.18   | 19.49  |
| **Molecule Retrieval**  |                |                     |               |                 |         |        |
| MoleculeSTM      | 65.27          | 59.95               | 72.39         | 54.57           | 60.17   | 62.03  |
| MoMu             | 63.60          | 56.34               | 70.52         | 53.27           | 59.23   | 57.75  |
| Llama3-8b        | 20.60          | 19.76               | 20.67         | 20.07           | 20.90   | 22.46  |
| Llama2-7b        | 20.58          | 20.43               | 20.03         | 20.77           | 21.49   | 21.39  |
| Galactica-125m   | 21.62          | 28.04               | 21.57         | 20.92           | 20.09   | 31.02  |
| Galactica-1.3b   | 22.17          | 29.64               | 22.45         | 21.67           | 19.71   | 31.02  |
| Galactica-6.7b   | 22.30          | 30.44               | 22.60         | 22.22           | 19.28   | 33.16  |
| Molt5-large      | 23.54          | 39.79               | 23.89         | 23.36           | 18.18   | 41.18  |
| Molt5-large-c2s  | 23.00          | 32.31               | 23.86         | 21.32           | 19.87   | 29.95  |
| Random           | 20.28          | 20.43               | 21.12         | 19.84           | 20.58   | 19.


### Citing the Dataset
Please cite this dataset using the following BibTeX entry:

```bibtex
@misc{moltextqa,
	author = {Siddhartha Laghuvarapu, Namkyeong Lee, Chufan Gao, Jimeng Sun},
	title = {MolTextQA},
	year = 2024,
	url = {https://huggingface.co/datasets/sl160/MolTextQA},
	doi = {10.57967/hf/2443},
	publisher = {Hugging Face}
}
