from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
CORPUS_PATH = BASE / 'parsed_corpus.json'
CONFLICTS_PATH = BASE / 'contradictions.json'

corpus: List[Dict[str, Any]] = []
conflicts: List[Dict[str, Any]] = []
embedder = None
matrix = None
backend = None
tfidf_aux = None
conflict_cache = []
stem_vocab = set()
stemmer = PorterStemmer()
STOPWORDS = set('a an the and or to of for on in is are was were be been being what when where how who why can could should would does do did if i my me we you your their his her our this that these those with from by about into over under at as it its on all each any a student students minimum maximum much many may must than then there here'.split())


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text


def tokens(text: str) -> List[str]:
    return re.findall(r'[a-z0-9%]+', text.lower())


class Passage(BaseModel):
    id: str
    source: str
    source_type: str
    page: Optional[int] = None
    section_ref: str
    title: str = ''
    text: str
    similarity_score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: List[Passage]
    retrieval_backend: str
    decision_score: float


def build_index() -> None:
    global corpus, conflicts, embedder, matrix, backend
    if not CORPUS_PATH.exists():
        raise FileNotFoundError('parsed_corpus.json not found. Run build_corpus.py first.')
    if not CONFLICTS_PATH.exists():
        raise FileNotFoundError('contradictions.json not found. Run build_corpus.py first.')

    corpus = json.loads(CORPUS_PATH.read_text(encoding='utf-8'))
    conflicts = json.loads(CONFLICTS_PATH.read_text(encoding='utf-8'))
    documents = [clean_text(f"{x.get('title','')} {x['section_ref']} {x['text']}") for x in corpus]

    # Use semantic embeddings when sentence-transformers is installed.
    # TF-IDF is a deterministic fallback so the project remains runnable locally.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        matrix = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
        backend = 'sentence-transformers:all-MiniLM-L6-v2'
    except Exception as exc:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(documents)
        embedder = vectorizer
        backend = f'tfidf-fallback ({type(exc).__name__})'
    # Auxiliary TF-IDF is also kept for robust coverage/abstention decisions.
    global tfidf_aux, stem_vocab
    tfidf_aux = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    tfidf_aux.fit(documents)
    stem_vocab = {stemmer.stem(v) for v in tfidf_aux.vocabulary_}
    # Precompute contradiction evidence vectors once; do not re-encode them per request.
    global conflict_cache
    conflict_cache = []
    for conflict in conflicts:
        clause_items = []
        valid = True
        for clause in conflict['clauses']:
            item = corpus_doc(clause['source'], clause['section_ref']) if corpus else None
            if item is None:
                valid = False; break
            clause_items.append(item)
        if not valid:
            continue
        evidence_text = ' '.join([conflict['topic']] + conflict.get('topics', []) + conflict.get('signature', []))
        evidence_text += ' ' + ' '.join(x['text'] for x in clause_items)
        conflict_cache.append({
            'conflict': conflict,
            'items': clause_items,
            'evidence_text': evidence_text,
            'evidence_vec': None,
            'clause_vecs': [],
        })
    if backend.startswith('sentence-transformers'):
        alltexts=[]
        for c in conflict_cache:
            alltexts.append(c['evidence_text']); alltexts.extend(x['text'] for x in c['items'])
        vecs=embedder.encode(alltexts, convert_to_numpy=True, normalize_embeddings=True) if alltexts else []
        k=0
        for c in conflict_cache:
            c['evidence_vec']=vecs[k]; k+=1
            c['clause_vecs']=list(vecs[k:k+len(c['items'])]); k += len(c['items'])
    else:
        for c in conflict_cache:
            c['evidence_vec']=tfidf_aux.transform([c['evidence_text']])
            c['clause_vecs']=[tfidf_aux.transform([x['text']]) for x in c['items']]


def encode_query(question: str):
    if backend.startswith('sentence-transformers'):
        return embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    return embedder.transform([question])


def similarity_scores(query_vector) -> np.ndarray:
    return cosine_similarity(query_vector, matrix)[0]


def corpus_doc(source: str, section_ref: str) -> Optional[Dict[str, Any]]:
    for item in corpus:
        if item['source'] == source and item['section_ref'] == section_ref:
            return item
    return None


def make_passage(item: Dict[str, Any], score: float) -> Passage:
    return Passage(
        id=item['id'],
        source=item['source'],
        source_type=item.get('source_type', 'unknown'),
        page=item.get('page'),
        section_ref=item['section_ref'],
        title=item.get('title', ''),
        text=clean_text(item['text']),
        similarity_score=round(float(score), 4),
    )


def lexical_coverage(question: str, text: str) -> float:
    q = set(tokens(question))
    t = set(tokens(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def weighted_key_coverage(question: str, evidence: str) -> float:
    """Measures coverage of distinctive query terms, not generic question words."""
    if tfidf_aux is None:
        return lexical_coverage(question, evidence)
    vocab = tfidf_aux.vocabulary_
    idf_map = dict(zip(tfidf_aux.get_feature_names_out(), tfidf_aux.idf_))
    q_terms = []
    for term in tokens(question):
        if len(term) < 3 or term in STOPWORDS:
            continue
        stem = stemmer.stem(term)
        # Keep important terms even when the exact surface form differs.
        best_idf = idf_map.get(term, 0.0)
        if best_idf > 0:
            q_terms.append((term, best_idf))
        elif stem in stem_vocab:
            q_terms.append((term, 1.0))
        else:
            # Unknown distinctive words (e.g. "wedding") count strongly against coverage.
            q_terms.append((term, 3.0))
    if not q_terms:
        return 0.0
    ev_tokens = set(tokens(evidence))
    covered = 0.0
    total = 0.0
    for term, weight in q_terms:
        total += weight
        if term in ev_tokens or any(stemmer.stem(x) == stemmer.stem(term) for x in ev_tokens):
            covered += weight
    return covered / total if total else 0.0


def missing_distinctive_terms(question: str, evidence: str) -> List[str]:
    """Return high-information question terms absent from the candidate evidence."""
    if tfidf_aux is None:
        return []
    idf_map = dict(zip(tfidf_aux.get_feature_names_out(), tfidf_aux.idf_))
    ev_tokens = set(tokens(evidence))
    missing=[]
    for term in tokens(question):
        if len(term) < 4 or term in STOPWORDS:
            continue
        if term in ev_tokens or any(stemmer.stem(x)==stemmer.stem(term) for x in ev_tokens):
            continue
        weight = idf_map.get(term, 4.0 if term not in idf_map else idf_map[term])
        if weight >= 3.5:
            missing.append(term)
    return sorted(set(missing))


def signature_coverage(question: str, signature: List[str]) -> float:
    q_terms=set(tokens(question)); hits=0
    for concept in signature:
        ct=tokens(concept)
        if ct and all(stemmer.stem(y) in {stemmer.stem(x) for x in q_terms} for y in ct):
            hits += 1
    return hits / max(1, len(signature))


def required_group_hits(question: str, groups: List[List[str]]) -> int:
    qset=set(tokens(question)); stems={stemmer.stem(x) for x in qset}; hits=0
    for group in groups:
        found=False
        for concept in group:
            cterms=tokens(concept)
            if cterms and all(stemmer.stem(ct) in stems for ct in cterms):
                found=True; break
        hits += int(found)
    return hits


def has_numeric_or_temporal_evidence(question: str, evidence: str) -> bool:
    q = question.lower(); e = evidence.lower()
    nums = re.findall(r'\b\d+(?:\.\d+)?%?\b', q)
    number_words = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
    q_words = [w for w in number_words if re.search(rf'\b{w}\b', q)]
    if nums and not any(n in e for n in nums):
        return False
    if q_words and not any(w in e for w in q_words):
        return False
    if any(p in q for p in ['how early', 'how long', 'deadline', 'how many days', 'when must']) and not re.search(r'\b\d+(?:\.\d+)?\b|\b(?:day|days|week|weeks|month|months|year|years|august|november)\b', e):
        return False
    return True


def intent_evidence_ok(question: str, evidence: str) -> bool:
    q = question.lower(); e = evidence.lower()
    if any(p in q for p in ['can i ', 'can a ', 'can an ', 'are students allowed', 'is there a formal appeal', 'is it allowed', 'may a student']):
        return any(p in e for p in ['allowed', 'permitted', 'eligible', 'may ', 'can ', 'option to', 'procedure', 'permission'])
    if any(p in q for p in ['procedure', 'how do i', 'how can i', 'how to apply', 'process for']):
        return any(p in e for p in ['procedure', 'process', 'apply', 'application', 'submit', 'request', 'form'])
    return has_numeric_or_temporal_evidence(question, evidence)


def conflict_match(question: str, top_indices: np.ndarray, scores: np.ndarray, q_vec) -> Optional[Tuple[Dict[str, Any], List[Passage], float]]:
    """Evidence-driven conflict detection using registered clause pairs, not question keywords."""
    best = None
    for cached in conflict_cache:
        conflict = cached['conflict']
        clause_items = cached['items']
        if backend.startswith('sentence-transformers'):
            topic_score = float(cosine_similarity(q_vec, cached['evidence_vec'].reshape(1, -1))[0][0])
            clause_scores = [float(cosine_similarity(q_vec, v.reshape(1, -1))[0][0]) for v in cached['clause_vecs']]
        else:
            topic_score = float(cosine_similarity(q_vec, cached['evidence_vec'])[0][0])
            clause_scores = [float(cosine_similarity(q_vec, v)[0][0]) for v in cached['clause_vecs']]
        min_clause_score = min(clause_scores)
        coverage = weighted_key_coverage(question, cached['evidence_text'])
        sig_cov = signature_coverage(question, conflict.get('signature', conflict.get('topics', [])))
        group_hits = required_group_hits(question, conflict.get('required_groups', [])) if conflict.get('required_groups') else 0
        combined_raw = 0.55 * topic_score + 0.45 * min_clause_score
        combined = 0.55 * combined_raw + 0.25 * coverage + 0.20 * sig_cov

        threshold = 0.36 if backend.startswith('sentence-transformers') else 0.25
        if combined >= threshold and coverage >= 0.35 and sig_cov >= 0.20 and (not conflict.get('required_groups') or group_hits >= conflict.get('min_group_hits', 1)) and min_clause_score >= (0.12 if backend.startswith('sentence-transformers') else 0.07):
            passages = [make_passage(item, sc) for item, sc in zip(clause_items, clause_scores)]
            for idx in top_indices:
                if corpus[idx]['id'] not in {p.id for p in passages}:
                    passages.append(make_passage(corpus[idx], float(scores[idx])))
                    break
            candidate = (conflict, passages[:3], combined)
            if best is None or combined > best[2]:
                best = candidate
    return best

def retrieval_bonus(question: str, text: str) -> float:
    q = re.sub(r'\s+', ' ', question.lower()).strip()
    e = re.sub(r'\s+', ' ', text.lower()).strip()
    # Exact multi-word anchors are strong evidence even when the embedding score is modest.
    bonus = 0.0
    for phrase in re.findall(r'[a-z0-9%]+(?:\s+[a-z0-9%]+){1,4}', q):
        if len(phrase.split()) >= 2 and phrase in e:
            bonus = max(bonus, 0.12)
    return bonus


def decide_answer(question: str, top_indices: np.ndarray, scores: np.ndarray) -> Tuple[str, str, List[Passage], float]:
    q_vec = encode_query(question)
    conflict = conflict_match(question, top_indices, scores, q_vec)
    if conflict:
        c, citations, decision_score = conflict
        lines = [
            f"CONFLICT: The corpus contains incompatible provisions about {c['topic'].lower()}.",
            c['description'],
            '',
        ]
        for i, clause in enumerate(c['clauses'], start=1):
            lines.append(
                f"Rule {i}: {clause['section_ref']} ({clause['source']}) "
                f"— {clause['anchor']}."
            )
        lines.append('Because both provisions are part of the corpus and apply to the same question, the system does not choose one as the single answer.')
        return 'CONFLICT', '\n'.join(lines), citations, decision_score

    # Distinguish answerable from merely adjacent. The score is combined with query-term coverage.
    candidate_indices = np.argsort(np.array([float(scores[i]) + retrieval_bonus(question, corpus[i]['text']) for i in range(len(corpus))]))[::-1][:8]
    top = candidate_indices
    support = []
    for idx in top:
        lexical = weighted_key_coverage(question, corpus[idx]['text'])
        bonus = retrieval_bonus(question, corpus[idx]['text'])
        # Evidence similarity says the passage is about the query; weighted coverage says it
        # actually contains the distinctive concepts the user asked about.
        support_score = 0.60 * float(scores[idx]) + 0.30 * lexical + bonus
        support.append((support_score, idx, lexical))
    support.sort(reverse=True)
    threshold = 0.38 if backend.startswith('sentence-transformers') else 0.18
    aggregate_blob = ' '.join(corpus[idx]['text'] for _, idx, _ in support[:5])
    aggregate_lexical = weighted_key_coverage(question, aggregate_blob)
    aggregate_missing = missing_distinctive_terms(question, aggregate_blob)
    aggregate_intent = intent_evidence_ok(question, aggregate_blob)
    passing=[]
    for item in support:
        sup, idx, lex = item
        evidence=corpus[idx]['text']
        no_policy_signal=any(p in evidence.lower() for p in ['does not establish', 'no separate policy', 'not specified', 'not covered', 'does not state'])
        if sup >= threshold and lex >= 0.25 and intent_evidence_ok(question, evidence) and not no_policy_signal:
            passing.append(item)
    answerable = bool(passing) and aggregate_lexical >= 0.40 and aggregate_intent and not aggregate_missing
    chosen = passing[0] if passing else support[0]
    best_support, best_idx, best_lexical = chosen
    citations = [make_passage(corpus[idx], float(scores[idx])) for _, idx, _ in support[:3]]

    if not answerable:
        answer = (
            "NOT COVERED: The corpus does not contain a provision that directly answers this question. "
            "The passages shown are the closest retrieved evidence, not an answer."
        )
        return 'NOT_COVERED', answer, citations, best_support

    # Extractive synthesis: never invent facts not present in the evidence.
    selected_indices=[]
    for _, idx, _ in passing[:2] if passing else support[:2]:
        if idx not in selected_indices: selected_indices.append(idx)
    selected = [corpus[idx] for idx in selected_indices]
    answer_parts = []
    for item in selected:
        title = item.get('title') or item['section_ref']
        answer_parts.append(f"{title}: {clean_text(item['text'])}")
    answer = 'ANSWERED: The rulebook states:\n\n' + '\n\n'.join(answer_parts)
    return 'ANSWERED', answer, citations, best_support


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    print(f'Loaded {len(corpus)} corpus chunks using {backend}.')
    yield


app = FastAPI(title='The Rulebook That Argues With Itself', version='2.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok', 'chunks': len(corpus), 'contradictions': len(conflicts), 'backend': backend}


@app.post('/ask', response_model=AskResponse)
def ask_question(req: AskRequest):
    if matrix is None or not corpus:
        raise HTTPException(status_code=500, detail='Corpus index not initialized.')

    qv = encode_query(req.question)
    scores = similarity_scores(qv)
    top_indices = np.argsort(scores)[::-1][:8]
    status, answer, citations, decision_score = decide_answer(req.question, top_indices, scores)

    return AskResponse(
        status=status,
        answer=answer,
        citations=citations,
        retrieval_backend=backend,
        decision_score=round(float(decision_score), 4),
    )


STATIC = BASE / 'static'
if STATIC.exists():
    app.mount('/', StaticFiles(directory=str(STATIC), html=True), name='static')
