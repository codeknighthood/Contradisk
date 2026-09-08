from __future__ import annotations
import json
from pathlib import Path
import requests

BASE=Path(__file__).parent
TESTS=[BASE/'tests'/'answered.json',BASE/'tests'/'conflicts.json',BASE/'tests'/'not_covered.json']
URL='http://127.0.0.1:8000/ask'

rows=[]
for fp in TESTS:
    data=json.loads(fp.read_text(encoding='utf-8'))
    for t in data:
        r=requests.post(URL,json={'question':t['question']},timeout=30)
        r.raise_for_status(); out=r.json()
        rows.append({**t,'actual_status':out['status'],'ok':out['status']==t['expected_status']})

correct=sum(x['ok'] for x in rows)
print(f'Overall: {correct}/{len(rows)} = {correct/len(rows):.1%}')
for label in ['ANSWERED','CONFLICT','NOT_COVERED']:
    sub=[x for x in rows if x['expected_status']==label]
    good=sum(x['ok'] for x in sub)
    print(f'{label}: {good}/{len(sub)} = {good/len(sub):.1%}')

(Path(BASE/'evaluation_results.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
