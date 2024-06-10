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


def get_bert_score(hyps, refs):
    from bert_score import score

    print("************Calculating BertScore*************")
    P, R, F = score(hyps, refs, lang="en", verbose=True)
    return {
        "bert_p": P.mean().tolist(),
        "bert_r": R.mean().tolist(),
        "bert_f": F.mean().tolist(),
    }
