#!/usr/bin/env python3
"""
intent_expansion_pipeline.py

Intent Expansion Pipeline for "AI Workflow Analyst Intern" assignment.

Features:
- Loads inputs_for_assignment.json (intent_mapper + customer_messages)
- Vectorizes messages (Gemini embeddings -> sentence-transformers -> TF-IDF+SVD fallback)
- Clusters messages (HDBSCAN preferred, KMeans fallback)
- Analyzes clusters for alignment with existing primary/secondary intents
- Uses deterministic heuristics to propose:
    - splits of existing secondary intents
    - new secondary intents
  with quantitative justification and guardrails.
- Optional LLM assistance (Gemini or OpenAI) to generate clean intent IDs, display names,
  descriptions and guardrails for proposals.
- Produces outputs:
    - suggested_intents.json
    - intent_expansion_report.csv
    - cluster_examples.csv

Usage:
    python intent_expansion_pipeline.py --input inputs_for_assignment.json.txt --outdir out \
        --use_gemini --gemini_api_key YOUR_KEY

If you don't have a Gemini key, run without --use_gemini; it will still run deterministically.

Author: (your name)
"""

import os
import json
import re
import argparse
import logging
import uuid
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Optional external libs
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBT = True
except Exception:
    HAS_SBT = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# Optional LLM backends
HAS_GENAI = False
HAS_OPENAI = False
GENAI = None
OPENAI = None
try:
    import google.generativeai as genai  # pip install google-generativeai
    HAS_GENAI = True
    GENAI = genai
except Exception:
    HAS_GENAI = False

try:
    import openai  # pip install openai
    HAS_OPENAI = True
    OPENAI = openai
except Exception:
    HAS_OPENAI = False

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------
# Utilities
# ---------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_input(path: str) -> Dict:
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


# ---------------------------
# Vectorization (hierarchy of backends)
# ---------------------------

def embed_with_gemini(texts: List[str], model: str = "models/text-embedding-001", genai_client=None) -> np.ndarray:
    """
    Use Google Gemini embedding via google.generativeai (genai).
    genai_client: module object from google.generativeai (already configured)
    """
    if genai_client is None:
        raise RuntimeError("genai client not provided")
    # Batch in chunks to be safe
    EMB_BATCH = 50
    embs = []
    for i in range(0, len(texts), EMB_BATCH):
        batch = texts[i:i+EMB_BATCH]
        # genai.embed_content returns list/array depending on client; adapt as needed
        resp = genai_client.embeddings.create(model=model, input=batch)
        # resp -> contains 'data' list with 'embedding'
        for item in resp.data:
            embs.append(np.array(item.embedding, dtype=float))
    return np.vstack(embs)


def embed_with_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb


def tfidf_svd_vectors(texts: List[str], max_features: int = 10000, n_components: int = 64):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X_tfidf = vect.fit_transform(texts)
    n_comp = min(n_components, max(1, X_tfidf.shape[1]-1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X = svd.fit_transform(X_tfidf)
    return X, vect, svd, X_tfidf


def vectorize_texts(texts: List[str],
                     prefer_gemini: bool = False,
                     genai_client=None,
                     use_sentence_transformers: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Returns: embeddings ndarray (n_samples, d), metadata dict
    """
    meta = {}
    # Try Gemini embeddings
    if prefer_gemini and HAS_GENAI and genai_client is not None:
        try:
            logging.info("Embedding with Gemini...")
            emb = embed_with_gemini(texts, model="models/text-embedding-001", genai_client=genai_client)
            meta['method'] = 'gemini'
            return emb, meta
        except Exception as e:
            logging.warning("Gemini embedding failed: %s", e)

    # Try sentence-transformers
    if use_sentence_transformers and HAS_SBT:
        try:
            logging.info("Embedding with sentence-transformers...")
            emb = embed_with_sentence_transformers(texts)
            meta['method'] = 'sentence-transformers'
            return emb, meta
        except Exception as e:
            logging.warning("sentence-transformers embedding failed: %s", e)

    # Fallback TF-IDF + SVD
    logging.info("Using TF-IDF + SVD fallback for vectorization...")
    X, vect, svd, X_tfidf = tfidf_svd_vectors(texts, max_features=10000, n_components=64)
    meta['method'] = 'tfidf_svd'
    meta['vect'] = vect
    meta['svd'] = svd
    meta['X_tfidf'] = X_tfidf
    return X, meta


# ---------------------------
# Clustering
# ---------------------------

def cluster_embeddings(X: np.ndarray, prefer_hdbscan: bool = True, min_cluster_size: int = 8) -> Tuple[np.ndarray, dict]:
    """
    Returns labels array (n_samples,) and meta dict.
    If HDBSCAN is available we use it; otherwise KMeans with silhouette scan fallback.
    """
    meta = {}
    n = X.shape[0]
    # HDBSCAN
    if prefer_hdbscan and HAS_HDBSCAN:
        try:
            logging.info("Clustering with HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max( int(min_cluster_size), int(0.01*n) ),
                                        metric='euclidean', cluster_selection_method='eom')
            labels = clusterer.fit_predict(X)
            meta['method'] = 'hdbscan'
            meta['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
            return labels, meta
        except Exception as e:
            logging.warning("HDBSCAN failed: %s", e)

    # KMeans with heuristic k selection
    logging.info("Clustering with KMeans (silhouette-based search)...")
    best_labels = None
    best_k = None
    # reasonable range
    max_k = min(20, max(2, n//5))  # for 200 messages, up to ~20
    best_score = -1.0
    for k in range(2, max_k+1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labs = km.fit_predict(X)
            # silhouette requires at least 2 clusters
            from sklearn.metrics import silhouette_score
            sc = silhouette_score(X, labs)
            if sc > best_score:
                best_score = sc
                best_labels = labs
                best_k = k
        except Exception:
            continue
    if best_labels is None:
        # fallback to k= max(2, min(10,n))
        k = max(2, min(10, n))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        best_labels = km.fit_predict(X)
        best_k = k
    meta['method'] = 'kmeans'
    meta['n_clusters'] = best_k
    meta['silhouette'] = best_score
    return np.array(best_labels), meta


# ---------------------------
# Cluster analysis helpers
# ---------------------------

def top_terms_for_cluster(cluster_inds: List[int], vect: Optional[TfidfVectorizer], X_tfidf=None, top_n: int = 8) -> List[str]:
    """
    If vect and X_tfidf available, compute mean tf-idf in cluster and return top terms.
    Fallback: simple token frequency.
    """
    if vect is not None and X_tfidf is not None and len(cluster_inds) > 0:
        try:
            sub = X_tfidf[cluster_inds]
            mean_scores = np.asarray(sub.mean(axis=0)).ravel()
            top_idx = mean_scores.argsort()[::-1][:top_n]
            terms = [vect.get_feature_names_out()[i] for i in top_idx]
            return terms
        except Exception:
            pass
    # fallback token frequency
    return []


def representative_examples(cluster_inds: List[int], embeddings: np.ndarray, texts: List[str], n_examples: int = 5) -> List[str]:
    if len(cluster_inds) == 0:
        return []
    if embeddings is None:
        return [texts[i] for i in cluster_inds[:n_examples]]
    inds = np.array(cluster_inds)
    cluster_embs = embeddings[inds]
    centroid = cluster_embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(cluster_embs, centroid).ravel()
    order = sims.argsort()[::-1][:n_examples]
    chosen = [texts[inds[i]] for i in order]
    return chosen


# ---------------------------
# Heuristics for proposing intents
# ---------------------------

def propose_from_cluster(cluster_meta: dict,
                         existing_secondary_counts: Dict[str, int],
                         total_messages: int,
                         thresholds: dict) -> Optional[dict]:
    """
    Decide whether to propose new/split intent for the cluster.
    Returns proposal dict or None.
    Heuristics:
      - If cluster size < min_cluster_size -> no propose
      - If cluster has dominant existing secondary (>= split_ratio of the secondary's messages) and cluster coherence -> propose split
      - If cluster has no dominant existing secondary and cluster size >= new_cluster_pct -> propose new secondary
      - else -> no proposal (or mark for manual review)
    """
    size = cluster_meta['size']
    if size < thresholds['min_cluster_size']:
        return None

    sec_dist = cluster_meta.get('secondary_dist', {})  # dict -> counts inside cluster
    # Most common existing secondary and its count in this cluster
    if sec_dist:
        most_common_sec, most_common_count = max(sec_dist.items(), key=lambda x: x[1])
    else:
        most_common_sec, most_common_count = None, 0

    # If cluster represents a large chunk of a particular existing secondary's total messages
    prop_of_sec = 0.0
    if most_common_sec and existing_secondary_counts.get(most_common_sec, 0) > 0:
        prop_of_sec = most_common_count / existing_secondary_counts[most_common_sec]

    # cluster proportion of dataset
    cluster_pct = size / total_messages

    # Coherence metric (mean_sim)
    mean_sim = cluster_meta.get('mean_sim', 0.0)

    # Decision rules
    # 1) Split secondary: enough concentration from existing secondary (prop_of_sec >= T_split_ratio) and coherent
    if most_common_sec and prop_of_sec >= thresholds['T_split_ratio'] and mean_sim >= thresholds['coherence_min']:
        return {
            'proposal_type': 'split_secondary',
            'split_from_secondary': most_common_sec,
            'cluster': cluster_meta['cluster'],
            'reason': f"{most_common_count} msgs in cluster are from {most_common_sec} which is {prop_of_sec:.1%} of that secondary's messages; cluster coherence {mean_sim:.2f}."
        }

    # 2) New secondary: cluster large and not dominated by any existing secondary (most_common_prop < 0.5) and coherent
    most_common_prop = (most_common_count / size) if size > 0 else 0.0
    if most_common_prop < thresholds['dominance_threshold'] and cluster_pct >= thresholds['T_new_cluster_pct'] and mean_sim >= thresholds['coherence_min']:
        return {
            'proposal_type': 'new_secondary',
            'cluster': cluster_meta['cluster'],
            'reason': f"Cluster size {size} ({cluster_pct:.2%} of dataset); no dominant existing secondary (top hit {most_common_prop:.1%}); coherence {mean_sim:.2f}."
        }

    # 3) Large but low coherence -> manual review
    if cluster_pct >= thresholds['T_manual_review_pct'] and mean_sim < thresholds['coherence_min']:
        return {
            'proposal_type': 'manual_review',
            'cluster': cluster_meta['cluster'],
            'reason': f"Large cluster ({size} msgs, {cluster_pct:.2%}) with low coherence ({mean_sim:.2f}). Recommend manual review."
        }

    return None


# ---------------------------
# LLM-assisted naming & descriptions
# ---------------------------

def gemini_summarize_intent(representative_messages: List[str],
                            top_terms: List[str],
                            genai_client,
                            model: str = "models/gpt-4o-mini",
                            max_tokens: int = 300) -> dict:
    """
    Use Gemini (google.generativeai) to produce:
      - intent_id (snake_case)
      - display_name
      - description (1-2 sentences)
      - guardrails (list of 3)
    Returns dict with fields; on failure, returns deterministic fallback.
    """
    if genai_client is None:
        raise RuntimeError("genai client not provided")
    prompt = (
        "You are an expert conversational AI taxonomy designer. "
        "Given the following representative user messages and top terms, propose a short secondary-intent id (snake_case, <=30 chars), "
        "a display name (Title Case), a 1-2 sentence description, and 3 guardrail rules (short). "
        "Return valid JSON with keys: id, display_name, description, guardrails.\n\n"
        f"Top terms: {', '.join(top_terms[:20])}\n\n"
        "Representative messages:\n" + "\n".join(f"- {m}" for m in representative_messages[:8])
    )
    try:
        resp = genai_client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=0.0
        )
        text = ""
        # the response structure may vary by client version
        if hasattr(resp, "output") and resp.output:
            # new client: resp.output_text or resp.output
            try:
                # some clients have text in resp.output[0].content[0].text
                text = resp.output[0].content[0].text
            except Exception:
                try:
                    text = resp.output_text
                except Exception:
                    text = str(resp)
        else:
            text = str(resp)

        # try extract JSON
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            js = json.loads(m.group(0))
            return {
                'id': js.get('id'),
                'display_name': js.get('display_name'),
                'description': js.get('description'),
                'guardrails': js.get('guardrails') or js.get('rules') or []
            }
        else:
            # fallback parse heuristics
            return {'id': None, 'display_name': None, 'description': text.strip(), 'guardrails': []}
    except Exception as e:
        logging.warning("Gemini summarization failed: %s", e)
        return {'id': None, 'display_name': None, 'description': None, 'guardrails': []}


def openai_summarize_intent(representative_messages: List[str],
                            top_terms: List[str],
                            openai_client,
                            model: str = "gpt-4o-mini",
                            max_tokens: int = 300) -> dict:
    """
    Use OpenAI-style API to produce same outputs. Fallback deterministic on failure.
    """
    prompt = (
        "You are an expert conversational AI taxonomy designer. "
        "Given the following representative user messages and top terms, propose a short secondary-intent id (snake_case, <=30 chars), "
        "a display name (Title Case), a 1-2 sentence description, and 3 guardrail rules (short). "
        "Return valid JSON with keys: id, display_name, description, guardrails.\n\n"
        f"Top terms: {', '.join(top_terms[:20])}\n\n"
        "Representative messages:\n" + "\n".join(f"- {m}" for m in representative_messages[:8])
    )
    try:
        resp = openai_client.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens
        )
        text = resp['choices'][0]['message']['content']
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            js = json.loads(m.group(0))
            return {
                'id': js.get('id'),
                'display_name': js.get('display_name'),
                'description': js.get('description'),
                'guardrails': js.get('guardrails') or js.get('rules') or []
            }
        else:
            return {'id': None, 'display_name': None, 'description': text.strip(), 'guardrails': []}
    except Exception as e:
        logging.warning("OpenAI summarization failed: %s", e)
        return {'id': None, 'display_name': None, 'description': None, 'guardrails': []}


# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(input_path: str,
                 outdir: str,
                 use_gemini: bool = False,
                 gemini_api_key: Optional[str] = None,
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None,
                 prefer_sentence_transformers: bool = True,
                 prefer_hdbscan: bool = True,
                 config: Optional[dict] = None) -> dict:
    """
    Main pipeline runner. Returns a results dict.
    """
    if config is None:
        config = {}

    # Create outdir
    os.makedirs(outdir, exist_ok=True)

    # Configure Gemini / OpenAI clients if requested
    genai_client = None
    if use_gemini:
        if not HAS_GENAI:
            logging.error("Gemini (google.generativeai) package not installed. Install 'google-generativeai'.")
            raise RuntimeError("Gemini client required but not installed")
        if not gemini_api_key:
            raise RuntimeError("Gemini API key required when --use_gemini is set")
        # configure
        GENAI.configure(api_key=gemini_api_key)
        genai_client = GENAI

    openai_client = None
    if use_openai:
        if not HAS_OPENAI:
            logging.error("openai package not installed. Install 'openai'.")
            raise RuntimeError("OpenAI client required but not installed")
        if not openai_api_key:
            raise RuntimeError("OpenAI API key required when --use_openai is set")
        OPENAI.api_key = openai_api_key
        openai_client = OPENAI

    data = load_input(input_path)
    intent_mapper = data.get('intent_mapper', [])
    messages = data.get('customer_messages', [])

    # Build message dataframe
    rows = []
    for i, m in enumerate(messages):
        history = m.get('history', '') or ''
        current = m.get('current_human_message') or m.get('current_message') or ''
        text = clean_text((history + " " + current).strip())
        rows.append({
            'message_id': m.get('message_id', str(uuid.uuid4())),
            'conversation_id': m.get('conversation_id', ''),
            'history': history,
            'current': current,
            'text': text,
            'raw': m
        })
    df = pd.DataFrame(rows)
    texts = df['text'].fillna('').tolist()
    total_messages = len(texts)
    logging.info("Loaded %d messages", total_messages)

    # Build lookup for existing secondary distribution if messages already have labels
    # The assignment file sometimes doesn't include labels per message. We'll still support if present.
    # We search raw fields for 'primary'/'secondary' keys.
    existing_secondary_counts = Counter()
    for r in df['raw']:
        sec = r.get('secondary') or r.get('secondary_intent') or r.get('intent') or None
        if sec:
            existing_secondary_counts[sec] += 1

    # Vectorize
    emb, vmeta = vectorize_texts(texts,
                                 prefer_gemini=use_gemini,
                                 genai_client=genai_client,
                                 use_sentence_transformers=prefer_sentence_transformers)

    # If TFIDF fallback was used, extract vectorizer & X_tfidf for top terms
    vect = None
    X_tfidf = None
    if isinstance(vmeta, dict) and vmeta.get('method') == 'tfidf_svd':
        vect = vmeta.get('vect')
        X_tfidf = vmeta.get('X_tfidf')

    # Clustering
    labels, cmeta = cluster_embeddings(emb, prefer_hdbscan=prefer_hdbscan, min_cluster_size=config.get('min_cluster_size', 8))
    df['cluster'] = labels.tolist()

    # Compute cluster-level analytics
    clusters = sorted(set(labels))
    cluster_rows = []
    for c in clusters:
        inds = [i for i, lab in enumerate(labels) if lab == c]
        size = len(inds)
        # existing primary/secondary distribution
        sec_list = []
        pri_list = []
        for idx in inds:
            raw = df.at[idx, 'raw']
            sec = raw.get('secondary') or raw.get('secondary_intent') or raw.get('intent') or None
            pri = raw.get('primary') or raw.get('primary_intent') or None
            if sec:
                sec_list.append(sec)
            if pri:
                pri_list.append(pri)
        sec_counts = dict(Counter(sec_list).most_common(10))
        pri_counts = dict(Counter(pri_list).most_common(10))
        top_terms = top_terms_for_cluster(inds, vect, X_tfidf, top_n=10) if vect is not None else []
        reps = representative_examples(inds, emb, texts, n_examples=5)
        # coherence (mean cosine sim to centroid)
        mean_sim = None
        try:
            if len(inds) > 0:
                cent = emb[inds].mean(axis=0, keepdims=True)
                sims = cosine_similarity(emb[inds], cent).ravel()
                mean_sim = float(np.mean(sims))
        except Exception:
            mean_sim = None

        cluster_rows.append({
            'cluster': int(c),
            'size': size,
            'size_pct': size / max(1, total_messages),
            'top_terms': top_terms,
            'repr_examples': reps,
            'primary_dist': pri_counts,
            'secondary_dist': sec_counts,
            'mean_sim': mean_sim
        })

    report_df = pd.DataFrame(cluster_rows).sort_values('size', ascending=False)
    report_csv_path = os.path.join(outdir, 'intent_expansion_report.csv')
    report_df.to_csv(report_csv_path, index=False, encoding='utf8')
    logging.info("Wrote cluster report to %s", report_csv_path)

    # Heuristic thresholds (configurable)
    thresholds = {
        'min_cluster_size': config.get('min_cluster_size', max(8, int(0.02 * total_messages))),  # default: 2% or 8
        'T_split_ratio': config.get('T_split_ratio', 0.20),  # 20% of that secondary's messages concentrated in cluster -> split
        'T_new_cluster_pct': config.get('T_new_cluster_pct', 0.03),  # cluster >= 3% of dataset -> consider new
        'dominance_threshold': config.get('dominance_threshold', 0.5),  # if top mapping < 50% it's not dominant
        'coherence_min': config.get('coherence_min', 0.45),  # mean_sim threshold for semantic coherence
        'T_manual_review_pct': config.get('T_manual_review_pct', 0.05)  # if cluster >=5% but low coherence -> manual review
    }

    # Build global existing secondary counts (from intent_mapper if per-message labels missing)
    if not existing_secondary_counts and intent_mapper:
        for p in intent_mapper:
            for s in p.get('secondary_intents', []):
                # we don't have counts, keep zeroes (heuristics will rely on per-message distribution)
                existing_secondary_counts[s.get('id')] += 0

    # Propose intents
    proposals = []
    for cmeta_row in cluster_rows:
        proposal = propose_from_cluster(cmeta_row, existing_secondary_counts, total_messages, thresholds)
        if not proposal:
            continue
        # Add extra metadata
        cluster_inds = [i for i, lab in enumerate(labels) if lab == cmeta_row['cluster']]
        top_terms = cmeta_row['top_terms'][:8] if cmeta_row['top_terms'] else []
        reps = cmeta_row['repr_examples']
        # LLM naming (optional)
        named = {'id': None, 'display_name': None, 'description': None, 'guardrails': []}
        if use_gemini and genai_client is not None:
            try:
                named = gemini_summarize_intent(reps, top_terms, genai_client)
            except Exception as e:
                logging.warning("Gemini naming failed: %s", e)
                named = {'id': None, 'display_name': None, 'description': None, 'guardrails': []}
        elif use_openai and openai_client is not None:
            try:
                named = openai_summarize_intent(reps, top_terms, openai_client)
            except Exception as e:
                logging.warning("OpenAI naming failed: %s", e)
                named = {'id': None, 'display_name': None, 'description': None, 'guardrails': []}
        # deterministic fallback naming
        if not named.get('id'):
            # create deterministic snake_case id from top_terms
            cleaned_terms = []
            for t in top_terms[:4]:
                t = t.lower().strip()
                t = re.sub(r'[^a-z0-9]+', '_', t)
                t = t.strip('_')
                if t:
                    cleaned_terms.append(t)
            if not cleaned_terms:
                cleaned_terms = [f"intent_{uuid.uuid4().hex[:6]}"]
            intent_id = "_".join(cleaned_terms)[:40]
            display_name = " ".join([t.replace("_", " ").title() for t in cleaned_terms])
            desc = f"User messages about: {', '.join(top_terms[:6])}."
            guardrails = [
                "Do not classify messages about order logistics or refunds as this intent unless explicit product question.",
                "If message contains multiple distinct requests, mark ambiguous and escalate for human review.",
                "If the message is only a greeting/acknowledgement, do not assign this intent."
            ]
            named = {'id': intent_id, 'display_name': display_name, 'description': desc, 'guardrails': guardrails}

        # Quantitative justification
        justification = {
            'cluster_size': cmeta_row['size'],
            'cluster_pct': cmeta_row['size_pct'],
            'top_secondary_counts': cmeta_row['secondary_dist'],
            'mean_sim': cmeta_row['mean_sim']
        }

        proposals.append({
            'proposal_type': proposal['proposal_type'],
            'cluster': cmeta_row['cluster'],
            'proposed_intent_id': named['id'],
            'proposed_display_name': named['display_name'],
            'description': named['description'],
            'guardrails': named.get('guardrails', []),
            'top_terms': top_terms,
            'representative_examples': reps,
            'justification': justification,
            'raw_reason': proposal.get('reason', '')
        })

    # Save proposals
    sug_path = os.path.join(outdir, 'suggested_intents.json')
    with open(sug_path, 'w', encoding='utf8') as f:
        json.dump(proposals, f, indent=2, ensure_ascii=False)
    logging.info("Wrote suggested intents to %s", sug_path)

    # Save cluster_examples.csv for manual review
    cluster_examples = []
    for row in report_df.to_dict(orient='records'):
        c = row['cluster']
        # find up to 10 messages for this cluster
        inds = [i for i, lab in enumerate(labels) if lab == c]
        for idx in inds[:10]:
            cluster_examples.append({
                'cluster': c,
                'message_id': df.at[idx, 'message_id'],
                'text': df.at[idx, 'text'],
                'primary_label_guess': df.at[idx, 'raw'].get('primary') or df.at[idx, 'raw'].get('primary_intent'),
                'secondary_label_guess': df.at[idx, 'raw'].get('secondary') or df.at[idx, 'raw'].get('secondary_intent') or df.at[idx, 'raw'].get('intent')
            })
    pd.DataFrame(cluster_examples).to_csv(os.path.join(outdir, 'cluster_examples.csv'), index=False, encoding='utf8')

    # Return everything
    results = {
        'report_df': report_df,
        'proposals': proposals,
        'outdir': outdir,
        'config': {
            'vectorization': vmeta,
            'clustering': cmeta,
            'thresholds': thresholds
        }
    }
    return results


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Intent Expansion Pipeline")
    parser.add_argument('--input', '-i', required=True, help="Path to inputs_for_assignment.json (or the uploaded .txt)")
    parser.add_argument('--outdir', '-o', default='out_intent_expansion', help="Output directory")
    parser.add_argument('--use_gemini', action='store_true', help="Use Gemini embeddings & LLM (requires --gemini_api_key)")
    parser.add_argument('--gemini_api_key', default=None, help="Gemini API key")
    parser.add_argument('--use_openai', action='store_true', help="Use OpenAI for LLM naming (requires --openai_api_key)")
    parser.add_argument('--openai_api_key', default=None, help="OpenAI API key")
    parser.add_argument('--no_sentence_transformers', action='store_true', help="Do not use sentence-transformers even if installed")
    parser.add_argument('--prefer_hdbscan', action='store_true', help="Prefer HDBSCAN for clustering (if installed)")
    parser.add_argument('--min_cluster_size', type=int, default=None, help="Minimum cluster size to consider proposals")
    parser.add_argument('--t_split_ratio', type=float, default=None, help="Ratio for split heuristic (e.g., 0.2)")
    parser.add_argument('--t_new_cluster_pct', type=float, default=None, help="Cluster pct threshold to propose new intent (e.g., 0.03)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Configure clients if needed
    if args.use_gemini:
        if not HAS_GENAI:
            logging.error("Gemini support requested but package not installed. Install google-generativeai.")
            raise SystemExit(1)
        GENAI.configure(api_key=args.gemini_api_key)
    if args.use_openai:
        if not HAS_OPENAI:
            logging.error("OpenAI support requested but package not installed. Install openai.")
            raise SystemExit(1)
        OPENAI.api_key = args.openai_api_key

    conf = {}
    if args.min_cluster_size:
        conf['min_cluster_size'] = args.min_cluster_size
    if args.t_split_ratio:
        conf['T_split_ratio'] = args.t_split_ratio
    if args.t_new_cluster_pct:
        conf['T_new_cluster_pct'] = args.t_new_cluster_pct

    results = run_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        use_gemini=args.use_gemini,
        gemini_api_key=args.gemini_api_key,
        use_openai=args.use_openai,
        openai_api_key=args.openai_api_key,
        prefer_sentence_transformers=(not args.no_sentence_transformers),
        prefer_hdbscan=args.prefer_hdbscan,
        config=conf
    )

    logging.info("Pipeline completed. Outputs in %s", results['outdir'])
