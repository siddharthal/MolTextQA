import os
# os.environ['HF_HOME'] = '/srv/local/data/chufan2/huggingface/' # Set the Huggingface cache directory
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
import torch
from transformers import AutoTokenizer, OPTForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, T5ForConditionalGeneration
# from transformers.integrations import HfDeepSpeedConfig
import datasets
from itertools import chain
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import argparse
import re
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM

from transformers import StoppingCriteria
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [1, 1]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def process_messy_list_string(x, sep=','):
    x = [re.sub(r'[\[\]\'\"]', '', i.strip()) for i in x.split(sep)] # clean random quotations and brackets
    x = [i for i in x if i] # remove empty strings
    return x

def join_string(options):
    return "\n".join([f"({i+1}) {str(option)}" for i, option in enumerate(options)])

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    block_size = 256
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="galactica3") # Which model to run
    # parser.add_argument("--gen_smiles", action='store_true') # Generate SMILES strings instead of qa
    parser.add_argument("--train", action='store_true') # train instead of evaluate
    parser.add_argument("--load_finetuned", action='store_true') # load finetune instead of zero shot
    # parser.add_argument("--testrun", action='store_true') # load finetuneminstead of zero shot
    args = parser.parse_args()

    save_epoch = 1000
    save_path = './outputs/'
    save_finetune_path = './finetuned_models/'
    qa_data_path = './new_labels/data_splits.csv'

    if args.run == "galactica-6.7b":
        run = 'facebook/galactica-6.7b'
    elif args.run == "galactica-1.3b":
        run = 'facebook/galactica-1.3b'
    elif args.run == "galactica-125m":
        run = 'facebook/galactica-125m'
    elif args.run == 'molt5-large':
        run = 'laituan245/molt5-large'
    elif args.run == 'molt5-large-smiles2caption':
        run = 'laituan245/molt5-large-smiles2caption'
    elif args.run == 'molt5-large-caption2smiles':
        run = 'laituan245/molt5-large-caption2smiles'

    finetuned_model_path = os.path.join(save_finetune_path, f"{run.split('/')[-1]}")
    output_path = os.path.join(save_path, f"{run.split('/')[-1]}_load_finetuned={args.load_finetuned}.json")
    if 'galactica' in run:
        output_path = os.path.join(save_path, f"{run.split('/')[-1]}_load_finetuned={args.load_finetuned}.json")

    # if args.load_finetuned == True:
    #     # finetuned_model_latest = os.path.join(finetuned_model_path, os.listdir(finetuned_model_path)[-1])
    #     finetuned_model_latest = finetuned_model_path
    #     print("Loading finetuned model from", finetuned_model_latest)
    print(run, args)

    # # === Prepare the evaluation data ===
    df = pd.read_csv(qa_data_path, dtype=str)
    df['options'] = df['options'].apply(lambda x: process_messy_list_string(x, sep=','))
    df['smiles_option'] = df['smiles_option'].apply(lambda x: process_messy_list_string(x, sep=' '))
    df['correct'] = df['correct'].astype(int)
    df['smiles_correct'] = df['smiles_correct'].apply(lambda x: int(float(x)))
    df['ans1'] = pd.NA
    df['ans2'] = pd.NA
    for i in range(len(df)):
        if len(df['options'][i]) > df['correct'][i]-1:     # select correct from d['options'] using d['correct'] as index
            df.loc[i, 'ans1'] = df['options'][i][df['correct'][i]-1]
        if len(df['smiles_option'][i]) > df['smiles_correct'][i]-1:
            df.loc[i, 'ans2'] = df['smiles_option'][i][df['smiles_correct'][i]-1]
    print("failed text qa", df['ans1'].isna().sum(), "failed smiles qa", df['ans2'].isna().sum())

    # === train or evaluate ===
    if 'molt5' in run:
        # https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline
        # https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/trainer#transformers.Seq2SeqTrainer
        # https://github.com/blender-nlp/MolT5?tab=readme-ov-file#finetuning-molt5-based-models
        tokenizer = AutoTokenizer.from_pretrained(run, model_max_length=512)
        if args.load_finetuned == True:
            # model = T5ForConditionalGeneration.from_pretrained(finetuned_model_path, device_map="auto")
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(finetuned_model_path, device_map="auto")
            if args.train == False:
                model = model.merge_and_unload()
        else:
            model = T5ForConditionalGeneration.from_pretrained(run, device_map="auto")

        df['text1'] = "Q: You are given a SMILES string of a molecule, a question about the molecule and a set of candidate options. Pick the best option.\n"+\
            "SMILES_string:"+df['SMILES']+"\n"+\
            "Question:"+df['question']+'\n'+\
            "Options:"+df['options'].apply(lambda x: join_string(x))
        df['text2'] = "Q: You are given a sentence describing a molecule. Choose the SMILES string that best describes the SMILES string.\n"+\
            "Sentence:"+df['sentence']+'\n'+\
            "Options:"+df['smiles_option'].apply(lambda x: join_string(x))
        if run=='laituan245/molt5-large':
            df2 = pd.DataFrame(np.concatenate([df['text1'], df['text2']]), columns=['sentence'])
            df2['ans'] = np.concatenate([df['ans1'], df['ans2']])
            df2['split'] = np.concatenate([df['split'], df['split']])
        elif run=='laituan245/molt5-large-smiles2caption':
            df2 = df.copy()
            df2['sentence'] = df2['text1']
            df2['ans'] = df2['ans1']
        elif run=='laituan245/molt5-large-caption2smiles':
            df2 = df.copy()
            df2['sentence'] = df2['text2']
            df2['ans'] = df2['ans2']
    
        if args.train == True: 
            df2 = df2[df2['split'].isin(['train', 'valid'])]
            df2 = df2.sample(frac=1).reset_index(drop=True).astype(str) # shuffle

            def preprocess_function(sample, padding="max_length"):
                # add prefix to the input for t5
                # inputs = ["summarize: " + item for item in sample["dialogue"]]
                inputs = sample['sentence']
                # tokenize inputs
                model_inputs = tokenizer(inputs, max_length=320, padding=padding, truncation=True)
                # Tokenize targets with the `text_target` keyword argument
                labels = tokenizer(text_target=sample["ans"], max_length=320, padding=padding, truncation=True)

                if padding == "max_length":
                    labels["input_ids"] = [
                        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                    ]
                model_inputs["labels"] = labels["input_ids"].copy()
                return model_inputs

            train_data = datasets.Dataset.from_pandas(df2[df2['split']=='train'].reset_index(drop=True))
            train_tokenized = train_data.map(preprocess_function, batched=True, num_proc=4)
            print(f"Keys of tokenized dataset: {list(train_tokenized.features)}")

            valid_data = datasets.Dataset.from_pandas(df2[df2['split']=='valid'].reset_index(drop=True))
            valid_tokenized = valid_data.map(preprocess_function, batched=True, num_proc=4)
            print(f"Keys of tokenized dataset: {list(valid_tokenized.features)}")

            lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # we want to ignore tokenizer pad token in the loss
            label_pad_token_id = -100
            # Data collator
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8)
            # Define training args
            training_args = Seq2SeqTrainingArguments(
                eval_strategy="epoch",
                output_dir=finetuned_model_path, 
                auto_find_batch_size=True,
                learning_rate=2e-5, # higher learning rate
                save_strategy="steps",
                save_steps=1000,
                num_train_epochs=2,
            )
            # Create Trainer instance
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_tokenized,
                eval_dataset=valid_tokenized,
            )
            # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

            trainer.train()
            trainer.save_model(finetuned_model_path)
        else:
            with torch.no_grad():
                df2 = df2[df2['split']=='test']
                answers = []
                errors = []
                steps = 0

                for text in tqdm(df2['sentence'].values):
                    if steps % save_epoch == 0: # Save every save_epoch steps
                        with open(output_path, "w") as f:
                            json.dump(answers, f)
                        if len(answers) > 0:
                            print(answers[-1])
                            print('===========')
                    try:
                        # outputs = pipe(text)
                        inputs = tokenizer(text, return_tensors="pt")
                        input_ids = inputs.input_ids.to(model.device)
                        attention_mask = inputs.attention_mask.to(model.device)
                        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=20)
                        answers.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
                    except Exception as e:
                        print(e)
                        answers.append("ERROR")
                        # errors.append(steps)
                        steps += 1
                        continue                        
                    steps += 1
                
                with open(output_path, "w") as f:
                    json.dump(answers, f)

                print(f"Errors: {len(errors)}")
                print(errors)

    elif 'galactica' in run:
        df['text1'] = "Q: You are given a SMILES string of a molecule, a question about the molecule and a set of candidate options. Pick the best option.\n"+\
            "SMILES_string:[START_I_SMILES]"+df['SMILES']+"[END_I_SMILES]\n"+\
            "Question:"+df['question']+'\n'+\
            "Options:"+df['options'].apply(lambda x: join_string(x))+'\n\nAnswer:'
        df['text2'] = "Q: You are given a sentence describing a molecule. Choose the SMILES string that best describes the SMILES string.\n"+\
            "Sentence:"+df['sentence']+'\n'+\
            "Options:"+df['smiles_option'].apply(lambda x: join_string(x))+'\n\nAnswer:'
        # for opt, add "</s>" to the end of the ans
        df['ans1'] = df['ans1'].apply(lambda x: x + "</s>")
        df['ans2'] = df['ans2'].apply(lambda x: x + "</s>")
        if args.train == True:        
            df2 = pd.DataFrame(np.concatenate([df['text1'] + df['ans1'].astype(str), df['text2'] + df['ans2'].astype(str)]), columns=['sentence'])
        else:
            df2 = pd.DataFrame(np.concatenate([df['text1'], df['text2']]), columns=['sentence'])
        df2['split'] = np.concatenate([df['split'], df['split']])

        tokenizer = AutoTokenizer.from_pretrained(run)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.eos_token = "</s>"
        if args.load_finetuned == True:
            # model = OPTForCausalLM.from_pretrained(finetuned_model_path, device_map="auto")
            model = AutoPeftModelForCausalLM.from_pretrained(finetuned_model_path, device_map="auto")
            if args.train == False:
                model = model.merge_and_unload()
            print("Loading finetuned model from", finetuned_model_path)
        else:
            model = OPTForCausalLM.from_pretrained(run, device_map="auto")
            print("Loading zero model", run)
        
        if args.train == True: 
            df2 = df2[df2['split'].isin(['train', 'valid'])]
            df2 = df2.sample(frac=1).reset_index(drop=True).astype(str) # shuffle
            print('nans', df2[df2['split']=='train'].isna().sum(), df2[df2['split']=='valid'].isna().sum())

            config = LoraConfig(r=16, lora_alpha=32, 
                                target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.05, bias="none", 
                                task_type="CAUSAL_LM")
            model = get_peft_model(model, config)
            print_trainable_parameters(model)

            train_data = datasets.Dataset.from_pandas(df2[df2['split']=='train'].reset_index(drop=True))
            train_tokenized_data = train_data.map(lambda x: tokenizer(x["sentence"]), batched=True, num_proc=4, remove_columns=list(df2.columns))
            train_lm_dataset = train_tokenized_data.map(group_texts, batched=True, num_proc=4)

            valid_data = datasets.Dataset.from_pandas(df2[df2['split']=='valid'].reset_index(drop=True))
            valid_tokenized_data = valid_data.map(lambda x: tokenizer(x["sentence"]), batched=True, num_proc=4, remove_columns=list(df2.columns))
            valid_lm_dataset = valid_tokenized_data.map(group_texts, batched=True, num_proc=4)
            
            # print(tokenized_squad)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            training_args = TrainingArguments(
                output_dir=finetuned_model_path,
                eval_strategy="epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
                push_to_hub=False,
                save_strategy="steps",
                save_steps=1000,
                num_train_epochs=2
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_lm_dataset,
                eval_dataset=train_lm_dataset,
                data_collator=data_collator,
            )
            trainer.train()
            trainer.save_model(finetuned_model_path)

        else:
            with torch.no_grad():
                df2 = df2[df2['split']=='test']
                # if args.testrun == True:
                #     df2 = df2.iloc[13924:]
                # pipe = pipeline("text-generation", model=finetuned_model_latest, device=0, max_new_tokens=100)
                answers = []
                errors = []
                steps = 0

                for text in tqdm(df2['sentence'].values):
                    if steps % save_epoch == 0: # Save every save_epoch steps
                        # if args.testrun == False:
                        with open(output_path, "w") as f:
                            json.dump(answers, f)
                        if len(answers) > 0:
                                print(answers[-1])
                                print('===========')
                    try:
                    # outputs = pipe(text)
                        inputs = tokenizer(text, return_tensors="pt", max_length=1024 - 100, truncation=True)
                        input_ids = inputs.input_ids.to(model.device)
                        # outputs = model.generate(input_ids, max_new_tokens=100, stopping_criteria=[EosListStoppingCriteria()])
                        outputs = model.generate(input_ids, max_new_tokens=20)
                        # outputs = tokenizer.decode(model.generate(input_ids, max_new_tokens=100, tokenizer=tokenizer, stop_strings=["</s>"])[0])
                        outputs = tokenizer.decode(outputs[0])
                        answers.append(outputs)
                    except Exception as e:
                        print(e)
                        answers.append("ERROR")
                        # errors.append(steps)
                        steps += 1
                        continue
                        
                    steps += 1
                
                # if args.testrun == False:
                with open(output_path, "w") as f:
                    json.dump(answers, f)

                print(f"Errors: {len(errors)}")
                print(errors)


# class QA_Datasets_SMILES_retrieval(torch.utils.data.Dataset):
#     def __init__(self, df):

#         self.SMILES_list, self.options_list, self.labels_list, self.questions = [], [], [], []
#         self.sentence_list = []
#         self.smiles_options_list = []
#         failures = []
#         for i in tqdm(range(len(df))):
#             self.SMILES_list.append(df["smiles"].values[i])
#             self.options_list.append(eval(df["options"].values[i]))
#             self.labels_list.append(int(df["correct_option"].values[i] - 1))
#             self.questions.append(df["question"].values[i])
#             self.sentence_list.append(df["sentence"].values[i])
#             self.smiles_options_list.append(df["smiles_options"].values[i])
            
#         print("Prepared data : {}".format(len(self.SMILES_list)))
#         print("Number of options: {}".format(np.mean([len(o) for o in self.options_list])))
    
#     def __getitem__(self, index):
#         return self.SMILES_list[index], self.questions[index], self.options_list[index], self.labels_list[index], \
#             self.sentence_list[index], self.smiles_options_list[index]
    
#     def __len__(self):
#         return len(self.SMILES_list)