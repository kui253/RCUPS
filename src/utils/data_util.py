import enum
from datasets import dataset_dict, load_dataset, DatasetDict
import re


def get_dataset(data_args, model_args):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        datasets = load_dataset(extension, data_files=data_files)

    return datasets


def data_preprocessing(datasets, tokenizer, training_args, data_args, model_args):
    # if data_args.save_dataset_path is None or data_args.reprocess_data:
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names

    text_column = data_args.text_column
    summary_column = data_args.summary_column

    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_v2_function(examples):
        ## examples:{str:[List[List]]}
        unit_utts = examples[text_column]  # List[List[str]]
        idxs = examples["selected_sents_bert"]
        targets = examples[summary_column]  #  List[str]

        bs = len(unit_utts)

        model_inputs = {
            "sorted_input_ids": [],
            "sorted_attention_mask": [],
            "sent_input_ids": [],
            "sent_attention_mask": [],
            "importance_label": [],
        }

        inputs = []

        for batch_idx in range(bs):
            if not data_args.in_val_and_test:
                utts = unit_utts[batch_idx].split("#")  # List[str]
                idxs[batch_idx] = sorted(idxs[batch_idx])
                utts2 = [utts[i] for i in idxs[batch_idx] if i < len(utts)]
                joint_utts = "<sep>".join(utts2)
                if len(utts) > data_args.max_utt_num:
                    utts = utts[: data_args.max_utt_num]
                ls = get_importance_ls(idxs[batch_idx], len(utts))
                assert len(ls) == len(utts), "importance label length error"
                model_inputs["importance_label"].append(ls)
                tokenized_utts_sent = tokenizer(
                    utts,
                    max_length=data_args.max_seq_len_per_utt,
                    padding=padding,
                    truncation=True,
                )
                model_inputs["sent_input_ids"].append(tokenized_utts_sent["input_ids"])
                model_inputs["sent_attention_mask"].append(
                    tokenized_utts_sent["attention_mask"]
                )
                tokenized_utts = tokenizer(
                    joint_utts,
                    max_length=data_args.max_source_length,
                    padding=padding,
                    truncation=True,
                )
                model_inputs["sorted_input_ids"].append(tokenized_utts["input_ids"])
                model_inputs["sorted_attention_mask"].append(
                    tokenized_utts["attention_mask"]
                )

            if "<sep>" not in tokenizer.additional_special_tokens:
                special_tokens_dict = {"additional_special_tokens": ["<sep>"]}
                tokenizer.add_special_tokens(special_tokens_dict)
            inputs_str = unit_utts[batch_idx]
            inputs_str = re.sub("#", "<sep>", inputs_str)

            inputs.append(inputs_str)
        baseline_input = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
        model_inputs["input_ids"] = baseline_input["input_ids"]
        model_inputs["attention_mask"] = baseline_input["attention_mask"]
        if data_args.in_val_and_test:
            model_inputs["sorted_input_ids"] = baseline_input["input_ids"]
            model_inputs["sorted_attention_mask"] = baseline_input["attention_mask"]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=data_args.max_target_length,
                padding=padding,
                truncation=True,
                add_special_tokens=False,
            )
            for k, v in labels.items():
                if k == "input_ids":
                    labels[k] = [x + [tokenizer.eos_token_id] for x in v]
                elif k == "attention_mask":
                    labels[k] = [x + [1] for x in v]

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]  # 没有传入attentionmask
        return model_inputs

    output_datasets = [None, None, None]

    map_function = preprocess_v2_function

    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[0] = train_dataset

    if training_args.do_eval:
        maping_prefix = "eval"
        # if not data_args.in_val_and_test:
        #     data_args.in_val_and_test = True
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[1] = eval_dataset

    if training_args.do_predict:
        maping_prefix = "predict"
        # if not data_args.in_val_and_test:
        #     data_args.in_val_and_test = True
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )
        predict_dataset = predict_dataset.map(
            map_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[2] = predict_dataset
    return output_datasets


def get_importance_ls(importance_label, num_utt):
    ret = [0] * num_utt
    for idx in importance_label:
        if idx < num_utt:
            ret[idx] = 1
    return ret
