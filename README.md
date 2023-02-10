# SWING 🏌️: Balancing Coverage and Faithfulness for Dialogue Summarization


**Authors**: Kung-Hsiang Huang ([khhuang3@illinois.edu](mailto:khhuang3@illinois.edu)), Siffi Singh, Xiaofei Ma, Wei Xiao, Feng Nan, Nicholas Dingwall, William Yang Wang, Kathleen McKeown .

## Dependencies
First, create a virtual environment and install depednencies specified in requirements.txt

```
conda create -n ds python=3.8
conda activate ds
pip install -r requirements.txt
```

Then, create separate enviroments for BARTScore and FactCC, following the instructions for [BARTScore](https://github.com/neulab/BARTScore) and [FactCC](https://github.com/salesforce/factCC).

## Data
The preprocessed data can be downloaded from [here](https://drive.google.com/drive/u/3/folders/1PRGuqv-FD5SmI3oyj90YKr_KVaJueFvw) (`dialogsum.zip` and `samsum.zip`). Please create a `data` folder and unzip these two files into this folder.


## Training
To train the model, run `train.py`. For example,
```
python train.py --exp_name $EXP_NAME --model_name facebook/bart-large --learning_rate 3e-5 --weight_decay 1e-3 --warmup_epoch 0 --accumulate_step 4 --batch_size 2   --dataset dialogsum  --use_nli --do_uncovered --do_invalid --uncovered_weights 0.7 --invalid_weights 0.2 --do_factcc_validate --do_gradient_checkpointing
```

Training parameters are specified in `args.py`. You can specify each the value of each args_key by passing `--args_key arg_value`. Below illustrate some of the important keys.

```
--max_sequence_length: Maximum input length. Dialogues longer than this length will be truncated.

--model_name: The name of the model to load from HugingFace.

--dataset: One of {dialogsum, samsum}.

--use_nli: Enable this will train a generator with the NLIBART class, which is also proposed model.

--do_invalid: Do contrastive learning. (Invalid loss is the name we gave in the early stage of the experiment)

--do_uncovered: Do uncovered loss.

--exp_name: Name of the experiment. The model checkpoint will be saved in `args.output_dir/args.exp_name`.

--data_dir: (Deprecated) Directory of the input data. Specifying --dataset would affect this parameter.

--use_robust: (Deprecated) Do MLE with adversarial training. This was used in the early stage of the experiment.

--do_factcc_validate: (Deprecated) Use FactCC to further validate the goodness of the generated summary. Not used in the final solution.

--do_factcc_uncovered: (Deprecated) Not used in the final solution.
```

The trained checkpoints can be found in [here](https://drive.google.com/drive/folders/1zeqppst-YJPvonN0TNruRqWwDc-rsuV6?usp=sharing) (`[dialogsum|samsum]_best/best.pt`) for research purposes.


## Evaluation

To run evaluation on trained models, execute the `test.py` script as follows:

```
 python test.py --checkpoint_path $PATH_TO_MODEL/best.pt
```

If you already have your generated summaries (e.g. our training script would produce a `$PATH_TO_OUTOUT/test_pred.json`), you can directly run the following command to avoid running inference again and save time.

```
python test_file.py --dataset samsum --output_file $PATH_TO_OUTOUT/test_pred.json
```


## Citation

```bibtex
@inproceedings{huang-etal-2023-swing,
    title = "SWING 🏌️: Balancing Coverage and Faithfulness for Dialogue Summarization",
    author = "Huang, Kung-Hsiang  and
      Singh, Siffi  and
      Ma, Xiaofei  and
      Xiao, Wei  and
      Nan, Feng  and
      Dingwall, Nicholas  and
      Wang, William Yang  and
      McKeown, Kathleen",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```