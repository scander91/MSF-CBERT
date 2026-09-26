"""Step 02a — Generate paraphrases for the minority attack-type classes (R2, D1).

Generator: Qwen/Qwen2.5-7B-Instruct, 4-bit NF4 (fits the 16 GB card) — named explicitly
in every artifact; this pipeline never calls it "GPT-4" (invariant 5).

Split-independence: paraphrases are generated ONCE for every event of the target
classes in the processed corpus. Per-split train-only containment happens later in
src/data/augmentation.py (G1), which drops any paraphrase whose source fell in that
split's val/test. So generation itself cannot leak: it never sees the manifests.

Output data/raw/paraphrases.jsonl, one object per paraphrase:
  {"source_eventid", "text", "label", "sim", "human_ok": 0, "generator"}
  sim = cosine(all-MiniLM-L6-v2(source), MiniLM(paraphrase)); the >=0.85 filter is
  applied downstream in augmentation.py, so rejected generations stay visible in the
  accounting. human_ok=0 means "not yet reviewed by the author".
"""
import argparse, json, os, time
import pandas as pd, torch
from src.util import load_config, log

PROMPT = (
    "Rewrite the following terrorism-incident description in different words for a "
    "text-classification dataset. Keep every factual detail exactly as stated — "
    "perpetrator, location, date, target, weapon, casualties — and a similar length. "
    "Do not add, remove, or speculate about facts. Output ONLY the rewritten text.\n\n"
    "Description: {}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap events (smoke tests only)")
    a = ap.parse_args(); cfg = load_config(a.config)
    p, ag = cfg["paths"], cfg["augmentation"]
    k = int(ag["paraphrases_per_event"])

    df = pd.read_parquet(p["processed"])
    src = df[df["label"].isin(ag["target_classes"])][["eventid", "label", "text"]] \
        .reset_index(drop=True)
    if a.limit:
        src = src.head(a.limit)
    log(f"generator={ag['generator']} | {len(src):,} source events x {k} = "
        f"{len(src) * k:,} paraphrases")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(ag["generator"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(ag["generator"], quantization_config=bnb,
                                                 device_map="auto")
    model.eval()

    from sentence_transformers import SentenceTransformer
    import numpy as np
    st = SentenceTransformer(ag["sim_encoder"], device="cuda")

    # resume support: skip sources already fully generated
    done = {}
    if os.path.exists(p["paraphrases"]):
        for line in open(p["paraphrases"]):
            r = json.loads(line)
            done[r["source_eventid"]] = done.get(r["source_eventid"], 0) + 1
        src = src[~src["eventid"].map(lambda e: done.get(str(e), 0) >= k)]
        log(f"resume: {len(done):,} sources already present; {len(src):,} to go")

    # long summaries first so OOM (if any) hits immediately, and batches stay uniform
    src = src.assign(_l=src["text"].str.len()).sort_values("_l", ascending=False)

    t0, written = time.time(), 0
    with open(p["paraphrases"], "a") as f:
        for s in range(0, len(src), a.batch_size):
            chunk = src.iloc[s:s + a.batch_size]
            msgs = [[{"role": "user", "content": PROMPT.format(t)}] for t in chunk["text"]]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, padding=True,
                                          return_tensors="pt", return_dict=True).to(model.device)
            max_new = min(512, int(enc["input_ids"].shape[1] * 1.25) + 80)
            with torch.no_grad():
                for gi in range(k):
                    out = model.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95,
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
                    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                    texts = [t.strip().strip('"') for t in texts]
                    e_src = st.encode(list(chunk["text"]), normalize_embeddings=True)
                    e_par = st.encode(texts, normalize_embeddings=True)
                    sims = (e_src * e_par).sum(axis=1)
                    for (_, row), t, sim in zip(chunk.iterrows(), texts, sims):
                        f.write(json.dumps({
                            "source_eventid": str(row["eventid"]), "text": t,
                            "label": row["label"], "sim": round(float(sim), 4),
                            "human_ok": 0, "generator": ag["generator"]}) + "\n")
                        written += 1
                    f.flush()
            if (s // a.batch_size) % 10 == 0:
                rate = written / max(1, time.time() - t0)
                log(f"{written:,} written | {rate:.2f} para/s | "
                    f"ETA {((len(src) * k - written) / max(rate, 0.01)) / 60:.0f} min")
    log(f"DONE: {written:,} paraphrases appended -> {p['paraphrases']}")


if __name__ == "__main__":
    main()
