from tqdm import tqdm
import torch


def get_rouge_score(hyps, refs):
    import rouge

    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=False,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    py_rouge_scores = evaluator.get_scores(hyps, refs)
    return py_rouge_scores


def find_largest_two_string_indices(lst):
    len_indices = [(len(s), i) for i, s in enumerate(lst)]
    largest_two = sorted(len_indices)[-2:]
    return [i for _, i in largest_two]


def get_sum_center_data(path, save_path=None, mode="train"):
    import json

    with open(path, "r") as f:
        data = json.load(f)
    pbar = tqdm(data, desc="extracting")
    for idx, example in enumerate(pbar):
        summary_pieces = example["summary"]
        utts = example["dialogue"]
        if len(utts) >= len(summary_pieces):
            divnum = len(utts) // len(summary_pieces)

        else:
            divnum = len(utts)
        if divnum > 2:
            divnum = 2
        idxs = []
        for id, piece in enumerate(summary_pieces):
            if id > len(utts):
                break
            repeat_piece = [piece] * len(utts)
            scores = get_rouge_score(repeat_piece, utts)
            f1 = [scores["rouge-1"][i]["f"][0] for i in range(len(scores["rouge-1"]))]
            tensor_list = torch.tensor(f1)
            result = torch.topk(tensor_list, divnum).indices
            idxs.extend(result.tolist())

        example["utts_idx_inorder"] = sorted(
            set(idxs) | set(find_largest_two_string_indices(utts))
        )
    save_path += "you_data_name_{}.json".format(mode)
    json.dump(data, open(save_path, "w"))
