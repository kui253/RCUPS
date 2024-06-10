import re
import time
import json

import requests

from tqdm import tqdm
import sys


out_dir, url, target_file, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

json_dir = "final_test.json"
SUMM_TASK = "You will receive a dialogue along with a summary of that dialogue, and you are required to assess the quality of the summary. The evaluation criteria are as follows: \
Faithfulness: Rate on a scale of 1-5. \
A score of 5 indicates that the content of the summary fully aligns with the information in the dialogue, with no errors. \
A score of 4 means that the majority of the content in the summary matches the dialogue, but there are minor discrepancies. \
A score of 3 suggests that some parts of the summary correspond to the dialogue, but there are some errors. \
A score of 2 indicates that most of the content in the summary does not align with the dialogue, with only a few points matching. \
A score of 1 means that the generated content is entirely incorrect. \
Fluency: Evaluate the smoothness of the language used. \
A score of 5 signifies that the summary is very fluent, with no linguistic errors or misuse of words. \
A score of 4 indicates a few errors, but they are minimal. \
A score of 3 suggests there are some errors. \
A score of 2 means the sentences are somewhat incoherent. \
A score of 1 indicates significant difficulties and complete incoherence. \
Informativeness: Assess the amount of information regarding time, characters, locations, and events or others in the dialogue. \
A score of 5 indicates that the summary contains a wealth of information, very comprehensive. \
A score of 4 means there is a good amount of information with a few omissions. \
A score of 3 suggests that some important information is missing. \
A score of 2 indicates that the summary contains only a small amount of information. \
A score of 1 means that there is absolutely no information. \
You are only required to output the scores for each criterion, adhering strictly to the following format: Faithfulness: x\n Fluency: x\n Informativeness: x\n, where x represents the score you marked, do not output any other content.\
"
with open(json_dir, "r") as f3:
    dialogus = json.load(f3)
if mode == "summ":
    datas = [i["summary"] for i in dialogus]
else:
    if mode == "txt":
        with open(target_file, "r") as f4:
            datas = f4.readlines()
    else:
        with open(target_file, "r") as f3:
            datas = json.load(f3)
            datas = [i[f"idx{n}"] for n, i in enumerate(datas)]

for iddx, task_sum in enumerate([SUMM_TASK]):
    all_result = []

    for n, section in tqdm(enumerate(dialogus)):
        sents = "\n".join(section["unit_utts"].split("#"))
        question_entity = (
            f"Dialogue: {sents}, corresponding abstract summary: {datas[n]}"
        )

        data = {
            "system_prompt": task_sum,
            "context_list": [question_entity],
        }
        time.sleep(0.5)
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
