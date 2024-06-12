# Training Galactica and MolT5

## Prerequisites

- Please ensure that the paths are correct in llm_qa.py, specifically, the path to the QA data
- Make sure to create directories to save the data to.
```jsx
mkdir outputs
```

## 1. Training Galactica

Please take a look at run_galactica.sh to familiarize yourself with the training process. You may wish to play around wtih these args. 

```jsx
CUDA_VISIBLE_DEVICES=0 bash run_galactica.sh
```

## 2. Training MolT5

Please take a look at run_molt5.sh to familiarize yourself with the training process. You may wish to play around wtih these args. 
```jsx
CUDA_VISIBLE_DEVICES=0 bash run_molt5.sh
```
## 3. Viewing Results

You can use results.ipynb to view the outputs and create a csv of the predictions.
