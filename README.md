# Mitigating Missing Information for Dialogue Summarization

**Author**: Steeve Huang (@khshuang, [khhuang3@illinois.edu](mailto:khhuang3@illinois.edu)).

## Abstract (not finalized)

We aim to reduce the factual inconsistency problem faced by text generation model. Specifically, we are interested in addressing the "missing information" challenge in abstractive summarization for dialogue. 

While prior work has achieved improvements on other categories of factual inconsistency errors, such as modality error and object error, missing information has remained unsolved. In this work, we aim to address this issue by providing finer-grainer training signals to the generators in addition to maximum likelihood estimation.

## Dependencies
First, create a virtual environment and install depednencies specified in requirements.txt

```
conda create -n ds python=3.7
conda activate ds
pip install -r requirements.txt
```

Then, create separate enviroments for BARTScore and FactCC, following the [BARTScore instruction](https://github.com/neulab/BARTScore) and [FactCC instruction](https://github.com/salesforce/factCC).

## Data
The preprocessed data can be downloaded from [here](https://s3.console.aws.amazon.com/s3/upload/khshuang-intern-data?region=us-west-2) (`dialogsum.zip` and `samsum.zip`). Please create a `data` folder and unzip these two files into this folder.


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

--do_invalid: Do contrastive learning. (Invalid loss is the name we gave in the early stage of the internship)

--do_uncovered: Do uncovered loss.

--exp_name: Name of the experiment. The model checkpoint will be saved in `args.output_dir/args.exp_name`.

--data_dir: (Deprecated) Directory of the input data. Specifying --dataset would affect this parameter.

--use_robust: (Deprecated) Do MLE with adversarial training. This was used in the early stage of the internship.

--do_factcc_validate: (Deprecated) Use FactCC to further validate the goodness of the generated summary. Not used in the final version.

--do_factcc_uncovered: (Deprecated) Not used in the final version.
```

The trained checkpoints can be found in [here](https://s3.console.aws.amazon.com/s3/buckets/khshuang-intern-data?region=us-west-2&tab=objects) (`*_checkpoint_best.zip`) for reproduction purposes.


## Evaluation

To run evaluation on trained models, execute the `test.py` script as follows:

```
 python test.py --checkpoint_path $PATH_TO_MODEL/best.pt
```

If you already have outputs generated (e.g. our training script would produce a `$PATH_TO_OUTOUT/test_pred.json`), you can directly run the following command to save time by avoiding running inference again.

```
python test_file.py --dataset samsum --output_file $PATH_TO_OUTOUT/test_pred.json
```


If there is any issue, please contact 
## References

- [Pre-approval ticket](https://issues.amazon.com/issues/SCIPUB1a-791)


