import json, re, math, itertools
from collections import defaultdict
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from rapidfuzz import fuzz, process


def extract_text_blocks(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
           
            t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text.append(t)
    whole = "\n".join(text)
   
    whole = re.sub(r"[ \t]+", " ", whole)
    
    whole = re.sub(r"(\w)-\n(\w)", r"\1\2", whole)
    return whole


DEBATE_HEADING_RE = re.compile(
    r"(?P<title>([A-Z][A-Za-z’'().:&/\- ]+?))(?:\n| )(?=\d+\.\s|\[?\d{6,}\]?|Topical Questions|Business of the House|Women|Access|Charities|Sports|Creative|Community|Poverty|House of Commons|Christians|Project Spire)",
    flags=re.MULTILINE,
)

SPEAKER_LINE_RE = re.compile(
    r"(?m)^(?P<name>[A-Z][A-Za-z .’'-\-]+(?:\([^)]+\))?(?:\s*\([A-Za-z/ \-]+\))?)\s*:\s"
)

def split_debates(raw_text):
  
    positions = [(m.start(), m.group("title").strip()) for m in DEBATE_HEADING_RE.finditer(raw_text)]
    if not positions:
        return [{"title": "Debate", "text": raw_text}]
    debates = []
    for idx, (start, title) in enumerate(positions):
        end = positions[idx+1][0] if idx+1 < len(positions) else len(raw_text)
        debates.append({"title": title, "text": raw_text[start:end].strip()})
    return debates

def parse_speeches(section_text):
    
    chunks = []
    indices = [m.start() for m in SPEAKER_LINE_RE.finditer(section_text)]
    if not indices:
        return []
    indices.append(len(section_text))
    for i in range(len(indices)-1):
        s = section_text[indices[i]:indices[i+1]]
        m = SPEAKER_LINE_RE.match(s)
        if not m:
            continue
        speaker = m.group("name").strip()
        speech = s[m.end():].strip()
        chunks.append({"speaker_raw": speaker, "speech_text": collapse_paragraphs(speech)})
    return chunks

def collapse_paragraphs(t):
   
    t = re.sub(r"\n{2,}", " <PARA> ", t)
    t = re.sub(r"\n", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


SPEAKER_META_RE = re.compile(r"^(?P<name>.+?)(?:\s*\((?P<constit>[^)]+)\))?(?:\s*\((?P<party>[^)]+)\))?$")

def parse_speaker_meta(s):
    m = SPEAKER_META_RE.match(s)
    if not m:
        return {"name": s}
    return {
        "name": m.group("name").strip(),
        "constituency": (m.group("constit") or "").strip() or None,
        "party": (m.group("party") or "").strip() or None,
    }



def build_ner():
    
    return pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1
    )

def clean_entities(ents):
    out = []
    for e in ents:
        txt = e["word"].strip()
        if not txt or len(txt) == 1 and not txt.isalnum():
            continue
        
        txt = re.sub(r"\s+/+\s+", "/", txt)
        out.append({"text": txt, "type": e["entity_group"], "score": float(e.get("score", 0.0))})
    return out


def try_build_re():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        mod = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        gen = pipeline("text2text-generation", model=mod, tokenizer=tok)
        return gen
    except Exception:
        return None

def extract_relations(gen, text, max_len=350):
   
    sents = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    triples = []
    if gen is None:
       
        for m in re.finditer(r"(?:what|which)\s+(?:steps|progress|support)[^?]*?(?:on|to|for)\s+([^?]+)\?", text, flags=re.I):
            topic = m.group(1).strip(" .;:)")
            triples.append(("MP", "asks_about", topic))
        return triples

    for chunk in sents:
        out = gen(chunk, max_length=256, do_sample=False)
        if not out: 
            continue
        text_out = out[0]["generated_text"]
        
        for subj, rel, obj in re.findall(r"<triplet>\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*</triplet>", text_out):
            triples.append((subj.strip(), rel.strip(), obj.strip()))
    return triples


def normalize_key(s):
    s = re.sub(r"[^\w\s&\-./]", "", s).strip().lower()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def build_entity_map(all_entities, sim_threshold=92):
    
    canonical = {}
    mentions = defaultdict(list)

    unique_texts = []
    for e in all_entities:
        unique_texts.append(e["text"])

    for txt in unique_texts:
        nk = normalize_key(txt)
        if nk in canonical:
            continue
      
        choices = list(canonical.keys())
        if choices:
            best, score, _ = process.extractOne(nk, choices, scorer=fuzz.token_sort_ratio)
            if score >= sim_threshold:
                canon = best
            else:
                canon = nk
        else:
            canon = nk
        canonical[nk] = canon

   
    out = {}
    for e in all_entities:
        nk = normalize_key(e["text"])
        canon = canonical.get(nk, nk)
        if canon not in out:
            out[canon] = {"canonical": e["text"], "type": e["type"], "mentions": [], "count": 0}
        out[canon]["mentions"].append(e["text"])
        out[canon]["count"] += 1
        
        if "score" in e and e.get("score", 0) > 0 and out[canon].get("best_score", -1) < e["score"]:
            out[canon]["type"] = e["type"]
            out[canon]["best_score"] = e["score"]
    
    for k in out:
        out[k]["mentions"] = sorted(set(out[k]["mentions"]))
        out[k].pop("best_score", None)
    return out


def process_pdf(pdf_path, out_path):
    text = extract_text_blocks(pdf_path)
    debates = split_debates(text)

    ner = build_ner()
    re_gen = try_build_re()

    final = {"debates": []}
    for d in debates:
        speeches = parse_speeches(d["text"])
        all_ents = []
        enriched = []
        for sp in speeches:
            meta = parse_speaker_meta(sp["speaker_raw"])
            ents = clean_entities(ner(sp["speech_text"])) if sp["speech_text"] else []
            rels = extract_relations(re_gen, sp["speech_text"])
            all_ents.extend(ents)
            enriched.append({
                "speaker": meta,
                "speech_text": sp["speech_text"],
                "entities": ents,
                "relations": [{"subject": s, "predicate": p, "object": o} for (s, p, o) in rels]
            })
        entity_map = build_entity_map(all_ents)
        final["debates"].append({
            "title": d["title"],
            "speeches": enriched,
            "entity_map": entity_map
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
  
    process_pdf("document.pdf", "output_enhanced.json")
