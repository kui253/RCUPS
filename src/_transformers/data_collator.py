import random
from re import X
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)


@dataclass
class MyDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # features List[dict]
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                feature["labels"] = (
                    feature["labels"] + remainder
                    if padding_side == "right"
                    else remainder + feature["labels"]
                )
        # if self.max_utt_threshold > 0:
        #     for feature in feature:
        #         for n in ['gt_input_ids','gt_attention_mask','']
        sent_features_ls = {}
        (
            sent_features_ls["len_ls"],
            sent_features_ls["sent_input_ids"],
            sent_features_ls["sent_attention_mask"],
        ) = pad_list_of_tensor(features, self.tokenizer.pad_token_id)
        max_sents_len = max(sent_features_ls["len_ls"])
        sent_features_ls["importance_label"] = softmax_sentlabel(
            features, max_sents_len, sent_features_ls["len_ls"]
        )
        normal_features_ls = [
            {
                "attention_mask": x["attention_mask"],
                "input_ids": x["input_ids"],
                "labels": x["labels"],
            }
            for x in features
        ]
        normal_padded_featurs = self.tokenizer.pad(
            normal_features_ls,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        temp_padded_features = [
            {
                "input_ids": f["sorted_input_ids"],
                "attention_mask": f["sorted_attention_mask"],
            }
            for f in features
        ]
        temp_padded_features = self.tokenizer.pad(
            temp_padded_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_features = {
            "input_ids": normal_padded_featurs["input_ids"],
            "attention_mask": normal_padded_featurs["attention_mask"],
            "labels": normal_padded_featurs["labels"],
            "sorted_input_ids": temp_padded_features["input_ids"],
            "sorted_attention_mask": temp_padded_features["attention_mask"],
            "sent_input_ids": sent_features_ls["sent_input_ids"],
            "sent_attention_mask": sent_features_ls["sent_attention_mask"],
            "importance_label": sent_features_ls["importance_label"],
            "len_ls": sent_features_ls["len_ls"],
        }
        padded_features = pad_other_features(
            padded_features, self.tokenizer.pad_token_id
        )

        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=padded_features["labels"]
            )
            padded_features["decoder_input_ids"] = decoder_input_ids

        return padded_features


def softmax_sentlabel(features, max_len, len_ls):
    for idx, sample in enumerate(features):
        after_softmax = torch.nn.functional.softmax(
            torch.tensor(sample["importance_label"], dtype=torch.float) / 0.5, dim=0
        )
        diff = max_len - len_ls[idx]
        pads = torch.tensor(diff * [0], dtype=torch.float)
        sample["importance_label"] = torch.concat([after_softmax, pads], dim=0).view(
            1, -1
        )

    return torch.cat([x["importance_label"] for x in features], dim=0)


def pad_to_same_dim(feature_list, pad_token_id, is_input_ids=False):
    max_len = min(max([len(x) for x in feature_list]), 512)
    if is_input_ids:
        all_attentions = []
    for idx in range(len(feature_list)):
        if len(feature_list[idx]) < max_len:
            feature_list[idx] += [pad_token_id] * (max_len - len(feature_list[idx]))
        else:
            feature_list[idx] = feature_list[idx][:max_len]
        if is_input_ids:
            if len(feature_list[idx]) < max_len:
                attention_mask = [1] * len(feature_list[idx]) + [0] * (
                    max_len - len(feature_list[idx])
                )
            else:
                attention_mask = [1] * max_len
            all_attentions.append(attention_mask)

    if not is_input_ids:
        return torch.tensor(feature_list)
    else:
        return torch.tensor(feature_list), torch.tensor(
            all_attentions, dtype=torch.bool
        )


def MyDataCollatorForBert(features):
    paded_features = {}
    paded_features["input_ids"], paded_features["attention_mask"] = pad_to_same_dim(
        [f["input_ids"] for f in features], 0, True
    )
    paded_features["token_type_ids"] = pad_to_same_dim(
        [f["tokenType_id"] for f in features], 0
    )
    for i, f in enumerate(features):
        f["clss_pos"] = [i for i in f["clss_pos"] if i < 512]
        f["labels"] = f["labels"][: len(f["clss_pos"])]
        if len(f["labels"]) != len(f["clss_pos"]):
            res = len(f["labels"]) - len(f["clss_pos"])
            if res < 0:
                f["clss_pos"] = f["clss_pos"][: len(f["labels"])]
        assert len(f["labels"]) == len(f["clss_pos"]), "wrong input ids dim"
    paded_features["labels"] = pad_to_same_dim([f["labels"] for f in features], 0)
    paded_features["clss_pos"] = pad_to_same_dim([f["clss_pos"] for f in features], -1)
    clss_mask = torch.ones(paded_features["clss_pos"].shape, dtype=torch.bool)
    clss_mask[paded_features["clss_pos"] == -1] = 0
    paded_features["clss_mask"] = clss_mask

    # paded_features["original_sent"] = [f["original_sent"] for f in features]

    # paded_features['summary_ids'],paded_features['summary_mask'] = pad_to_same_dim([f['summary'] for f in features],0,is_input_ids=True)
    return paded_features


def MyDataCollatorForBart(features):
    paded_features = {}
    paded_features["input_ids"], paded_features["attention_mask"] = pad_to_same_dim(
        [f["input_ids"] for f in features], 0, True
    )
    for i, f in enumerate(features):
        f["clss_pos"] = [i for i in f["clss_pos"] if i < 512]
        f["labels"] = f["labels"][: len(f["clss_pos"])]
        if len(f["labels"]) != len(f["clss_pos"]):
            res = len(f["labels"]) - len(f["clss_pos"])
            if res < 0:
                f["clss_pos"] = f["clss_pos"][: len(f["labels"])]
        assert len(f["labels"]) == len(f["clss_pos"]), "wrong input ids dim"
    paded_features["labels"] = pad_to_same_dim([f["labels"] for f in features], 0)
    paded_features["clss_pos"] = pad_to_same_dim([f["clss_pos"] for f in features], -1)
    clss_mask = torch.ones(paded_features["clss_pos"].shape, dtype=torch.bool)
    clss_mask[paded_features["clss_pos"] == -1] = 0
    paded_features["clss_mask"] = clss_mask

    return paded_features


def pad_other_features(features, pad_token_id):
    res_len = features["input_ids"].shape[1] - features["sorted_input_ids"].shape[1]
    if res_len > 0:
        features["sorted_input_ids"] = torch.cat(
            [
                features["sorted_input_ids"],
                torch.ones(
                    (features["sorted_input_ids"].shape[0], res_len), dtype=torch.long
                )
                * pad_token_id,
            ],
            dim=1,
        )
        features["sorted_attention_mask"] = torch.cat(
            [
                features["sorted_attention_mask"],
                torch.zeros(
                    (features["sorted_attention_mask"].shape[0], res_len),
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
    else:
        res_len = -res_len
        features["input_ids"] = torch.cat(
            [
                features["input_ids"],
                torch.ones((features["input_ids"].shape[0], res_len), dtype=torch.long)
                * pad_token_id,
            ],
            dim=1,
        )
        features["attention_mask"] = torch.cat(
            [
                features["attention_mask"],
                torch.zeros(
                    (features["attention_mask"].shape[0], res_len),
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
    return features


def pad_list_of_tensor(features, pad_token_id):
    """
    features: List[Dict[str,List[str]]
    """
    max_seq_len_in_a_batch = 0
    len_ls = []
    for sample in features:
        len_ls.append(len(sample["sent_input_ids"]))
        for utt in sample["sent_input_ids"]:
            l = len(utt)
            if l > max_seq_len_in_a_batch:
                max_seq_len_in_a_batch = l

    for sample in features:
        for k, v in sample.items():
            if k == "sent_input_ids":
                for utt in v:
                    diff = max_seq_len_in_a_batch - len(utt)
                    utt += [pad_token_id] * diff  # utt = utt + is wrong
            elif k == "sent_attention_mask":
                for mask in v:
                    diff = max_seq_len_in_a_batch - len(mask)
                    mask += [0] * diff
    return (
        len_ls,
        torch.cat([torch.tensor(x["sent_input_ids"]) for x in features], dim=0),
        torch.cat([torch.tensor(x["sent_attention_mask"]) for x in features], dim=0),
    )
