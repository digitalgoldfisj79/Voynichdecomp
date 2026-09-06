#!/usr/bin/env python3
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def make_loader(h):
    def load_ammerbach(root):
        reps=defaultdict(list)
        audit={"csv_files":[],"columns":{},"source_counts":{},"split_counts":{}}
        samples=[]; counts=Counter(); lengths=defaultdict(list)
        cutoffs={"train":500,"val":200,"test":500}
        expected={"train":1000,"val":400,"test":1000}
        for split in ("train","val","test"):
            matches=[x for x in root.rglob(f"{split}.csv") if ".ipynb_checkpoints" not in x.parts]
            if len(matches)!=1:
                raise RuntimeError(f"expected one canonical {split}.csv, found {matches}")
            csv_path=matches[0]; df=pd.read_csv(csv_path)
            if len(df)!=expected[split]:
                raise RuntimeError(f"{csv_path}: expected {expected[split]} rows, got {len(df)}")
            required={"file","top","bot","combined"}
            if not required.issubset(df.columns):
                raise RuntimeError(f"{csv_path}: missing {required-set(df.columns)}")
            audit["csv_files"].append({"path":str(csv_path),"rows":int(len(df)),"split":split})
            audit["columns"][str(csv_path)]=[str(x) for x in df.columns]
            audit["split_counts"][split]=int(len(df))
            for idx,row in df.reset_index(drop=True).iterrows():
                source="ammerbach_bookA" if idx<cutoffs[split] else "ammerbach_bookB"
                duration=h.tokenise_line(str(row["top"])); pitch=h.tokenise_line(str(row["bot"])); paired=h.tokenise_line(str(row["combined"]))
                flattened=[f"d{x}" for x in duration]+[f"p{x}" for x in pitch]
                ann={"pitch":pitch,"duration":duration,"paired":paired,"flattened":flattened}
                if not pitch or not paired:
                    raise RuntimeError(f"empty canonical annotation at {csv_path}:{idx}")
                counts[source]+=1; ident=f"{split}:{idx:04d}:{row['file']}"
                for rep,seq in ann.items():
                    if seq: reps[rep].append((source,seq,ident)); lengths[rep].append(len(seq))
                if len(samples)<12:
                    samples.append({"id":ident,"source":source,"lengths":{k:len(v) for k,v in ann.items()},"top_head":duration[:12],"bot_head":pitch[:12],"combined_head":paired[:12]})
        if sum(counts.values())!=2400 or counts!={"ammerbach_bookA":1200,"ammerbach_bookB":1200}:
            raise RuntimeError(f"Ammerbach balance failure: {counts}")
        audit.update({"annotations_parsed":2400,"source_counts":dict(counts),"checkpoint_files_excluded":len([x for x in root.rglob('*') if '.ipynb_checkpoints' in x.parts and x.is_file()]),"representation_lengths":{k:{"n":len(v),"mean":float(np.mean(v)),"median":float(np.median(v)),"min":min(v),"max":max(v)} for k,v in lengths.items()},"samples":samples,"book_identity_policy":"neutral bookA/bookB by documented balanced split halves; no chronology assigned"})
        return reps,audit
    return load_ammerbach
