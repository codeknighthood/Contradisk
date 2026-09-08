from __future__ import annotations
import csv, json, re
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / 'corpus'
CORPUS_DIR.mkdir(exist_ok=True)

PDF = ROOT / 'Student-Rule-Book.pdf'

SUPPLEMENT = CORPUS_DIR / 'demo_supplement.md'
FEE_TABLE = CORPUS_DIR / 'fee_deadlines.csv'

SUPPLEMENT.write_text(r'''# Demo University Student Rulebook Supplement

> **Corpus note:** This file is a synthetic supplement created for the project benchmark. It is not an official institutional circular. It exists to demonstrate evidence-backed conflict detection across multiple source formats.

## Academic Attendance

### §SUP-6.1 — Semester attendance requirement
A student must maintain **75% aggregate attendance in each semester** to be eligible to appear in the semester-end examination. Attendance is calculated from all courses registered during that semester. This clause applies to ordinary attendance eligibility.

### §SUP-6.2 — Academic-year attendance requirement
For semester-end examination eligibility, a student must maintain **75% aggregate attendance across the entire academic year**. Attendance is calculated from all courses taken during both semesters of the academic year. This clause is stated as the general attendance calculation rule.

### §SUP-6.3 — Attendance calculation records
Faculty members should maintain daily subject-wise attendance. Programme offices may publish attendance summaries periodically so that students can identify shortages before the examination period.

## Medical Attendance Exemption

### §SUP-7.1 — Medical relaxation
A student whose medical absence is supported by an approved medical certificate may be permitted to sit for the semester-end examination with **not less than 60% aggregate attendance**, subject to approval by the designated academic authority.

### §SUP-7.2 — Minimum floor for medical cases
Medical grounds may justify condonation of attendance shortage, but **no student, including a student with approved medical grounds, may appear in a semester-end examination with attendance below 70%**.

### §SUP-7.3 — Medical documentation
Medical certificates should identify the period of illness and be submitted through the programme office within the period notified for attendance shortage representations.

## Examination Applications

### §SUP-8.1 — Examination application
Students must submit the examination application through the institute using the prescribed process. The University issues the admit card for semester-end examinations.

### §SUP-8.2 — Admit card
A student must possess the University-issued admit card to appear in each semester-end examination paper.

## Fee Administration

### §SUP-F2 — Second-year fee deadline
The annual academic fee for a second-year student is due **by 15 August** each academic year unless a later date is specifically notified in writing.

### §SUP-F3 — Late payment
A student who does not pay by the due date may be charged late-payment charges according to the applicable fee schedule.

## Library

### §SUP-L1 — Standard loan period
Books issued through the library may normally be borrowed for seven days, subject to the library rules and availability of the material.

### §SUP-L2 — Overdue charges
Overdue books may attract a per-day fine in accordance with the applicable library rulebook. The exact rate is determined by the governing library provision.

## Student Conduct

### §SUP-D1 — Academic integrity
Copying, unauthorized collaboration, plagiarism, and other forms of academic misconduct are prohibited and may be referred for disciplinary action under the applicable rules.

### §SUP-D2 — Student explanation
A student involved in a disciplinary inquiry should be given an opportunity to explain the circumstances in writing before a decision is taken, subject to the governing procedure.

## Projects and AI tools

### §SUP-P1 — Project reports
Project reports must follow the formatting, submission, and evaluation instructions issued for the relevant programme and batch.

### §SUP-P2 — Use of AI tools
This supplement does not establish a separate policy specifying whether generative AI tools may be used to complete project reports. Students should refer to an applicable university or programme circular if one exists.
''', encoding='utf-8')

with FEE_TABLE.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['source','programme_year','fee_type','deadline','notes'])
    w.writeheader()
    rows = [
        ['fee_deadlines.csv','Second Year','Annual Academic Fee','20 August','Synthetic benchmark table; deliberately conflicts with §SUP-F2 for testing.'],
        ['fee_deadlines.csv','Third Year','Annual Academic Fee','20 August','Synthetic benchmark table.'],
        ['fee_deadlines.csv','First Year','Balance Academic Fee','As notified','Synthetic benchmark table.'],
        ['fee_deadlines.csv','Second Year','Security Deposit','At admission','Synthetic benchmark table.'],
        ['fee_deadlines.csv','Second Year','Reappear Examination Fee','Before application','Synthetic benchmark table.'],
        ['fee_deadlines.csv','Third Year','Reappear Examination Fee','Before application','Synthetic benchmark table.'],
    ]
    w.writerows([dict(zip(w.fieldnames, r)) for r in rows])


def normalize(s: str) -> str:
    s = s.replace('‐','-').replace('–','-').replace('—','-').replace('\u00ad','')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def parse_pdf(path: Path):
    reader = PdfReader(str(path))
    chunks = []
    for pageno, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ''
        text = normalize(raw)
        if not text:
            continue
        # Keep a page-level retrieval unit too. This catches headings/tables that are
        # not reliably reconstructed as clauses by PDF text extraction.
        chunks.append({
            'id': f'pdf-page-{pageno}', 'source': path.name, 'source_type': 'pdf',
            'page': pageno, 'section_ref': f'Page {pageno}', 'title': '', 'text': text
        })
        # Capture useful clause / heading markers inside each page.
        matches = list(re.finditer(r'(?<!\d)(\d+(?:\.\d+)+(?:\([a-z]\))?)\s+', text, flags=re.I))
        # Page-level fallback when no explicit clause marker is found.
        if not matches:
            chunks.append({
                'id': f'pdf-p{pageno}-001',
                'source': path.name,
                'source_type': 'pdf',
                'page': pageno,
                'section_ref': f'Page {pageno}',
                'title': '',
                'text': text,
            })
            continue
        # Split around explicit clause markers, while keeping enough context.
        starts = [m.start() for m in matches]
        for i, start in enumerate(starts):
            end = starts[i+1] if i+1 < len(starts) else len(text)
            body = text[start:end].strip()
            if len(body) < 35:
                continue
            ref = matches[i].group(1)
            # Further split lettered subclauses such as (a), (b), ... so citations are precise.
            parts = list(re.finditer(r'(?<![A-Za-z0-9])\(([a-z])\)\s+', body, flags=re.I))
            if parts:
                prefix = body[:parts[0].start()].strip()
                base_label = ref
                if prefix:
                    chunks.append({
                        'id': f'pdf-p{pageno}-{i+1:03d}-base', 'source': path.name, 'source_type': 'pdf',
                        'page': pageno, 'section_ref': base_label, 'title': '', 'text': prefix
                    })
                for j, pm in enumerate(parts):
                    pend = parts[j+1].start() if j+1 < len(parts) else len(body)
                    clause_body = body[pm.start():pend].strip()
                    if len(clause_body) >= 25:
                        letter = pm.group(1).lower()
                        chunks.append({
                            'id': f'pdf-p{pageno}-{i+1:03d}-{letter}', 'source': path.name, 'source_type': 'pdf',
                            'page': pageno, 'section_ref': f'{base_label}({letter})', 'title': '', 'text': clause_body
                        })
            else:
                chunks.append({
                    'id': f'pdf-p{pageno}-{i+1:03d}',
                    'source': path.name, 'source_type': 'pdf', 'page': pageno,
                    'section_ref': ref, 'title': '', 'text': body,
                })
    return chunks


def parse_markdown(path: Path):
    text = path.read_text(encoding='utf-8')
    chunks = []
    current = None
    lines = text.splitlines()
    for line in lines:
        m = re.match(r'^###\s+(§\S+)\s+—\s*(.*)$', line.strip())
        if m:
            if current and current['text'].strip():
                current['text'] = normalize(current['text'])
                chunks.append(current)
            ref = m.group(1).strip()
            title = (m.group(2) or '').strip()
            current = {
                'id': f'md-{ref.replace("§", "").replace(".", "-")}',
                'source': path.name,
                'source_type': 'markdown',
                'page': None,
                'section_ref': ref,
                'title': title,
                'text': '',
            }
        elif current is not None:
            current['text'] += ' ' + line
    if current and current['text'].strip():
        current['text'] = normalize(current['text'])
        chunks.append(current)
    return [c for c in chunks if len(c['text']) >= 25]


def parse_csv(path: Path):
    chunks = []
    with path.open(encoding='utf-8', newline='') as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            text = '; '.join(f'{k}: {v}' for k, v in row.items() if v)
            chunks.append({
                'id': f'csv-{path.stem}-{i}',
                'source': path.name,
                'source_type': 'csv',
                'page': None,
                'section_ref': f'CSV row {i}',
                'title': row.get('fee_type',''),
                'text': normalize(text),
            })
    return chunks


corpus = []
corpus.extend(parse_pdf(PDF))
corpus.extend(parse_markdown(SUPPLEMENT))
corpus.extend(parse_csv(FEE_TABLE))

# Remove exact duplicate texts but preserve separate source locations when they differ.
seen = set(); dedup = []
for c in corpus:
    key = (c['source'], c['section_ref'], c['text'])
    if key not in seen:
        seen.add(key); dedup.append(c)
corpus = dedup

(ROOT / 'parsed_corpus.json').write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding='utf-8')

contradictions = [
    {
        'id': 'C1',
        'topic': 'Attendance calculation period',
        'description': 'The corpus gives two different periods over which the 75% attendance requirement is calculated.',
        'clauses': [
            {'source': 'Student-Rule-Book.pdf', 'page': 6, 'section_ref': '6.1(a)', 'anchor': 'minimum attendance of 75%', 'expected_text_contains': 'Academic year'},
            {'source': 'demo_supplement.md', 'page': None, 'section_ref': '§SUP-6.1', 'anchor': '75% aggregate attendance in each semester', 'expected_text_contains': 'each semester'},
        ],
        'topics': ['attendance', '75 percent', '75%', 'semester', 'academic year', 'exam eligibility', 'attendance calculation']
    },
    {
        'id': 'C2',
        'topic': 'Medical attendance floor',
        'description': 'The synthetic supplement contains incompatible minimum attendance floors for approved medical cases.',
        'clauses': [
            {'source': 'demo_supplement.md', 'page': None, 'section_ref': '§SUP-7.1', 'anchor': 'not less than 60% aggregate attendance', 'expected_text_contains': '60%'},
            {'source': 'demo_supplement.md', 'page': None, 'section_ref': '§SUP-7.2', 'anchor': 'no student, including a student with approved medical grounds, may appear ... below 70%', 'expected_text_contains': 'below 70%'},
        ],
        'topics': ['medical', 'medical exemption', 'medical attendance', 'condonation', '60%', '70%', 'illness']
    },
    {
        'id': 'C3',
        'topic': 'Second-year annual fee deadline',
        'description': 'The synthetic markdown rule says 15 August while the fee table says 20 August.',
        'clauses': [
            {'source': 'demo_supplement.md', 'page': None, 'section_ref': '§SUP-F2', 'anchor': 'due by 15 August', 'expected_text_contains': '15 August'},
            {'source': 'fee_deadlines.csv', 'page': None, 'section_ref': 'CSV row 2', 'anchor': 'Annual Academic Fee; deadline: 20 August', 'expected_text_contains': '20 August'},
        ],
        'topics': ['second year fee', 'annual academic fee', 'fee deadline', 'payment deadline', '15 august', '20 august']
    },
]
(ROOT / 'contradictions.json').write_text(json.dumps(contradictions, indent=2, ensure_ascii=False), encoding='utf-8')

# Ground-truth evaluation data.
answered = [
    ['A01','What is the minimum pass percentage for each paper?','ANSWERED'],
    ['A02','How many grace marks can the University award?','ANSWERED'],
    ['A03','How long can a library book normally be issued for?','ANSWERED'],
    ['A04','What is the overdue library fine for the first seven days?','ANSWERED'],
    ['A05','How many credits does a BCA student need for the degree?','ANSWERED'],
    ['A06','How many years are allowed to complete the BCA programme?','ANSWERED'],
    ['A07','How long does a student have to apply for rechecking after results?','ANSWERED'],
    ['A08','Who conducts the semester-end examinations?','ANSWERED'],
    ['A09','Where are semester-end examinations conducted?','ANSWERED'],
    ['A10','What must a student possess to appear in each semester-end paper?','ANSWERED'],
    ['A11','How early does the Director have to announce students detained for attendance shortage?','ANSWERED'],
    ['A12','Can students repeat a failed semester-end course?','ANSWERED'],
    ['A13','What happens if a student is detained for attendance shortage?','ANSWERED'],
    ['A14','What percentage is the semester-end written examination worth for theory papers?','ANSWERED'],
    ['A15','What percentage is continuous evaluation worth for theory papers?','ANSWERED'],
    ['A16','Who handles internal examination complaints?','ANSWERED'],
    ['A17','What is the working time of the institute?','ANSWERED'],
    ['A18','Are bags and mobile phones allowed in the library?','ANSWERED'],
    ['A19','What happens to a lost or damaged library book?','ANSWERED'],
    ['A20','Can students use a computer lab outside their scheduled time?','ANSWERED'],
    ['A21','What happens after an academic year break?','ANSWERED'],
    ['A22','What is the maximum period for completing the programme?','ANSWERED'],
    ['A23','What is the pass mark for undergraduate programmes?','ANSWERED'],
    ['A24','How many academic year breaks are permissible?','ANSWERED'],
    ['A25','What is the penalty described for using unfair means in an examination?','ANSWERED'],
]
conflict_q = [
    ['C01','Over what period is the 75% attendance requirement calculated for exam eligibility?','CONFLICT'],
    ['C02','Is attendance calculated by semester or academic year?','CONFLICT'],
    ['C03','What is the attendance period used for the 75% requirement?','CONFLICT'],
    ['C04','A medically absent student has 65% attendance. Can the student appear for the exam?','CONFLICT'],
    ['C05','What is the minimum attendance allowed for a student with approved medical grounds?','CONFLICT'],
    ['C06','Does an approved medical certificate permit exam eligibility at 60% attendance?','CONFLICT'],
    ['C07','What is the deadline for a second-year student to pay the annual academic fee?','CONFLICT'],
    ['C08','Is the second-year annual fee due on 15 August or 20 August?','CONFLICT'],
    ['C09','Which source should a second-year student use for the annual fee deadline?','CONFLICT'],
    ['C10','The policy gives two different second-year fee deadlines. What does the corpus say?','CONFLICT'],
]
not_covered = [
    ['N01','What happens if I miss the semester-end examination because of my cousin’s wedding?','NOT_COVERED'],
    ['N02','Can I get a refund of unused money left on my campus food card?','NOT_COVERED'],
    ['N03','What is the exact grace period for submitting a late laboratory manual?','NOT_COVERED'],
    ['N04','Can students bring a personal room heater into the common room?','NOT_COVERED'],
    ['N05','What is the process for getting a duplicate degree certificate after water damage?','NOT_COVERED'],
    ['N06','How many internal grace marks are awarded for representing the state in cricket?','NOT_COVERED'],
    ['N07','What is the deadline for changing a hostel room during second year?','NOT_COVERED'],
    ['N08','Are vegetarian and non-vegetarian food prepared in separate kitchens?','NOT_COVERED'],
    ['N09','What fine applies to a library book overdue for exactly forty days?','NOT_COVERED'],
    ['N10','Can a BCA student take an elective from the BBA curriculum in semester IV?','NOT_COVERED'],
    ['N11','What is the formal procedure to change a Class Mentor?','NOT_COVERED'],
    ['N12','What alumni discount applies when booking the auditorium privately?','NOT_COVERED'],
    ['N13','How many compassionate-leave days are granted for a sibling’s marriage?','NOT_COVERED'],
    ['N14','What is the maximum laptop wattage allowed in the computer labs?','NOT_COVERED'],
    ['N15','Can a student appeal if a Proctor confiscates their mobile phone?','NOT_COVERED'],
    ['N16','What electrical safety specifications must hardware lab projects meet?','NOT_COVERED'],
    ['N17','Are academic fees fully refundable when a student leaves because of medical relocation?','NOT_COVERED'],
    ['N18','Does the rulebook permit ChatGPT for summer project reports?','NOT_COVERED'],
    ['N19','How do students obtain a bicycle parking permit?','NOT_COVERED'],
    ['N20','What compensation is paid if lab equipment is damaged by a power fluctuation?','NOT_COVERED'],
    ['N21','Can female students take overnight weekend leave during a cultural fest without a parental confirmation call?','NOT_COVERED'],
    ['N22','What happens if a student misses the monthly Parents-Institute Interaction meeting?','NOT_COVERED'],
    ['N23','Is there a sibling fee concession when two children are enrolled together?','NOT_COVERED'],
    ['N24','What is the maximum permitted hair length under the dress code?','NOT_COVERED'],
    ['N25','How are internal marks recalculated when a student misses a test to attend a national hackathon?','NOT_COVERED'],
]

tests_dir = ROOT / 'tests'; tests_dir.mkdir(exist_ok=True)
for name, rows in [('answered.json',answered),('conflicts.json',conflict_q),('not_covered.json',not_covered)]:
    data=[{'id':r[0],'question':r[1],'expected_status':r[2]} for r in rows]
    (tests_dir/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

word_count = sum(len(c['text'].split()) for c in corpus)
manifest = {
    'corpus_files': [PDF.name, SUPPLEMENT.name, FEE_TABLE.name],
    'mixed_formats': ['pdf','markdown','csv'],
    'word_count': word_count,
    'chunk_count': len(corpus),
    'contradiction_count': len(contradictions),
    'note': 'Three benchmark contradictions are explicitly registered in contradictions.json. The original PDF also contains independently observable inconsistencies.'
}
(ROOT/'corpus_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
