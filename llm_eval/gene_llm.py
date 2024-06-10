import time
import json

import requests
import nltk
from tqdm import tqdm
import sys


def eval_all(all_result, datas):
    # count words numbers
    count_preds = []
    count_labels = []
    for p, l in zip(all_result, datas):
        count_preds.append(len(p.strip().split()))
        count_labels.append(len(l.strip().split()))
    print(
        f"the avg length of preds is {sum(count_preds)/len(count_preds)} and label is {sum(count_labels)/len(count_labels)}"
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def get_rouge_score(hyps, refs):
        import rouge

        evaluator = rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=100,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True,
        )
        py_rouge_scores = evaluator.get_scores(hyps, refs)
        return py_rouge_scores

    all_result, datas = postprocess_text(all_result, datas)
    result = {}
    metrics_ls = [get_rouge_score]  # ,get_bert_score,get_meteor_score]
    for metrics in metrics_ls:
        res = metrics(all_result, datas)
        result.update(res)
    # keys: rouge-1(f,p,r),rouge-2,rouge-l,bert_p,bert_r,bert_f,meteor
    # Extract a few results from ROUGE
    result["rouge-1"] = result["rouge-1"]["f"] * 100
    result["rouge-2"] = result["rouge-2"]["f"] * 100
    result["rouge-l"] = result["rouge-l"]["f"] * 100

    result = {k: round(v, 4) for k, v in result.items()}
    print(result)


# "http://127.0.0.1:10003/qa_generate"
out_dir, url, json_dir = sys.argv[1], sys.argv[2], sys.argv[3]
SUMM_TASK = "Your Job is to summarize the given dialogue."
SUMM_TASK2 = (
    'Your Job is to summarize the given dialogue briefly in a format "who" "did" "what"'
)
SUMM_TASK3 = "Your Job is to summarize the given dialogue briefly."
with open(json_dir, "r") as f3:
    dialogus = json.load(f3)
summs = [i["summary"] for i in dialogus]


for iddx, task_sum in enumerate([SUMM_TASK, SUMM_TASK2, SUMM_TASK3]):
    all_result = []

    for n, section in tqdm(enumerate(dialogus)):
        split_sents = section["unit_utts"].split("#")
        sents = "\n".join(split_sents)
        idss = [split_sents[i] for i in section["label_sents"]]
        idss = "\n".join(idss)
        question_entity = (
            f"please summarize {sents}, and the utterances you need to focus on: {idss}"
        )

        data = {
            "system_prompt": task_sum,
            "context_list": [question_entity],
        }
        time.sleep(5)

        response_entity = requests.post(url, json=data)

        if response_entity.status_code == 200:
            json_data_entity = response_entity.json()
            answer_entity = json_data_entity["response"]

            all_result.append({f"idx{n}": answer_entity[0].strip() + "\n"})
        else:
            import time

            while response_entity.status_code != 200:
                print("Server no response, retrying...")
                time.sleep(60)
                response_entity = requests.post(url, json=data)

            # 发送POST请求
            json_data = response_entity.json()
            answer = json_data["response"]
            all_result.append({f"idx{n}": answer[0].strip() + "\n"})

    with open(str(iddx) + out_dir, "w") as f2:
        json.dump(all_result, f2)

    if len(all_result) == len(summs):
        all_result = [i[f"idx{n}"] for n, i in enumerate(all_result)]
        eval_all(all_result, summs)
    else:
        print("wrong length")


# run eval
