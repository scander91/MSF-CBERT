"""Quantitative attribution faithfulness over the test set.

Metrics (DeYoung et al., ERASER, ACL 2020):
  comprehensiveness = p(y|x) - p(y|x \\ top-k tokens)   — high = removing important tokens hurts
  sufficiency       = p(y|x) - p(y|top-k tokens only)   — low  = important tokens alone suffice
Importance = gradient × input on the embedding layer,
scored for N sampled test events. Replaces "confirms" language with numbers.
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from src.util import load_config, log, ensure_dir, read_manifest
from src.models.msf_cbert import MSFCBert
from src.features.verbalize import build_input_text


def grad_x_input(model, enc, device):
    emb_layer = model.encoder.get_input_embeddings()
    embs = emb_layer(enc["input_ids"].to(device)).detach().requires_grad_(True)
    out = model.encoder(inputs_embeds=embs,
                        attention_mask=enc["attention_mask"].to(device))
    a = out.last_hidden_state[:, 0]
    logits = model.classifier(model.text_head_proj(a)) if not model.use_graph \
        else model.classifier(model.fusion(a, torch.zeros(a.size(0),
                              model.rel_proj[model.relations[0]].out_features,
                              device=device)))
    y = logits.argmax(-1)
    logits.gather(1, y[:, None]).sum().backward()
    imp = (embs.grad * embs).sum(-1).abs().detach()             # (B, T)
    return imp, y


@torch.no_grad()
def prob_of(model, tok, texts, y, cfg, device):
    enc = tok(texts, truncation=True, padding=True,
              max_length=cfg["model"]["max_len"], return_tensors="pt").to(device)
    a = model.encode_text(**enc)
    logits = model.classifier(model.text_head_proj(a)) if not model.use_graph \
        else model.classifier(model.fusion(a, torch.zeros(a.size(0),
                              model.rel_proj[model.relations[0]].out_features,
                              device=device)))
    return torch.softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", required=True); ap.add_argument("--run", required=True)
    args = ap.parse_args(); cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(cfg["paths"]["runs_dir"], args.run, "metrics.json")))
    df = pd.read_parquet(cfg["paths"]["processed"])
    df = df[df["label"].isin(meta["labels"])]              # match the run's label scheme (D4)
    man = read_manifest(cfg, args.split)
    te = df.merge(man[man.split == "test"], on="eventid")
    te = te.sample(min(cfg["eval"]["faithfulness_sample"], len(te)), random_state=0)
    selected = json.load(open(f"{cfg['paths']['features_dir']}/selected_{args.split}.json"))["selected"]
    texts = [build_input_text(r, selected) for _, r in te.iterrows()]

    # Faithfulness is scored on the TEXT pathway (graph zeroed) so token attributions are
    # well-defined.
    model = MSFCBert(cfg, cfg["graph"]["relations"].keys(), len(meta["labels"]),
                     use_graph="nograph" not in args.run).to(device)
    model.load_state_dict(torch.load(
        os.path.join(cfg["paths"]["runs_dir"], args.run, "best.pt"), map_location=device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])

    comp, suff, k_frac, bs = [], [], cfg["eval"]["faithfulness_topk"], 16
    for s in range(0, len(texts), bs):
        batch = texts[s:s + bs]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=cfg["model"]["max_len"], return_tensors="pt")
        imp, y = grad_x_input(model, enc, device)
        p_full = prob_of(model, tok, batch, y, cfg, device)
        kept, removed = [], []
        for j, t in enumerate(batch):
            ids = enc["input_ids"][j]; mask = enc["attention_mask"][j].bool()
            scores = imp[j][mask]; toks = ids[mask]
            k = max(1, int(k_frac * len(toks)))
            top = torch.topk(scores, k).indices
            sel = torch.zeros(len(toks), dtype=torch.bool); sel[top] = True
            kept.append(tok.decode(toks[sel], skip_special_tokens=True))
            removed.append(tok.decode(toks[~sel], skip_special_tokens=True))
        comp += (p_full - prob_of(model, tok, removed, y, cfg, device)).tolist()
        suff += (p_full - prob_of(model, tok, kept, y, cfg, device)).tolist()

    res = {"run": args.run, "n": len(texts), "topk_frac": k_frac,
           "comprehensiveness_mean": float(np.mean(comp)),
           "sufficiency_mean": float(np.mean(suff))}
    ensure_dir(cfg["paths"]["tables_dir"])
    json.dump(res, open(f"{cfg['paths']['tables_dir']}/faithfulness_{args.run}.json", "w"), indent=2)
    log(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
