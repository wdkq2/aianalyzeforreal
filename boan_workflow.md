# 국토교통부 보도자료 임베딩 워크플로우
이 노트북은 공공데이터포털 OpenAPI를 이용해 보도자료 PDF를 수집하여 임베딩하고 검색 인덱스를 생성합니다.


```
SERVICE_KEY = 'YOUR_URL_ENCODED_SERVICE_KEY'
DCLSF_CD = 'A01'
START_DATE = '2020-01-01'
END_DATE = '2025-07-08'
PAGE_SIZE = 1000
DRIVE_DIR = '/content/drive/MyDrive/boan_data'

```


```
!pip install -q pdfplumber layoutparser[layoutmodels] sentence-transformers faiss-cpu
from google.colab import drive
import os, pathlib
if not pathlib.Path('/content/drive').exists():
    drive.mount('/content/drive')

```


```
import requests

BASE_URL = 'http://apis.data.go.kr/1613000/genFldPriorInfoDsc/getGenFldList'

def get_pdf_items():
    files = []
    page = 1
    while True:
        params = {
            'serviceKey': SERVICE_KEY,
            'pageNo': page,
            'numOfRows': PAGE_SIZE,
            'dclsfCd': DCLSF_CD,
            'startDate': START_DATE,
            'endDate': END_DATE,
            'viewType': 'json'
        }
        r = requests.get(BASE_URL, params=params)
        try:
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print('API 호출 오류:', e)
            print(r.text[:200])
            break
        for entry in data.get('data', []):
            for f in entry.get('attachFile', {}).get('file', []):
                url = f.get('downloadUrl')
                if url and url.lower().endswith('.pdf'):
                    files.append({'download_url': url,
                                   'regDate': entry.get('regDate'),
                                   'title': entry.get('title')})
        if data.get('currentCount', 0) < PAGE_SIZE:
            break
        page += 1
    return files

pdf_items = get_pdf_items()
print('총 다운로드 가능한 PDF 수:', len(pdf_items))

```


```
import pdfplumber, layoutparser as lp, tempfile

model = lp.models.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', extra_config=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5], label_map={0:'Text'})


paragraphs = []

for item in pdf_items:
    url = item.get('download_url')
    if not url:
        continue
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
        tmp.write(response.content)
        tmp.flush()
        with pdfplumber.open(tmp.name) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                layout = model.detect(page.to_image().original)
                for block in layout:
                    if block.type != 'Text':
                        continue
                    x0, y0, x1, y1 = block.coordinates
                    text = page.crop((x0, y0, x1, y1)).extract_text()
                    if text and 150 <= len(text) <= 500:
                        paragraphs.append({
                            'pdf_url': url,
                            'page': i,
                            'text': text.strip(),
                            'bbox': [x0, y0, x1, y1]
                        })

print('총 추출한 문단 수:', len(paragraphs))

```


```
import numpy as np, torch
from sentence_transformers import SentenceTransformer
import faiss

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('upskyy/e5-large-korean', device=device)
except Exception:
    model = SentenceTransformer('jhgan/ko-simcse-roberta-base')

texts = [p['text'] for p in paragraphs]
vecs = model.encode(texts, batch_size=32, convert_to_numpy=True)
faiss.normalize_L2(vecs)

```


```
import sqlite3, shutil, pathlib
pathlib.Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)
db_path = '/tmp/docs.db'
index_path = '/tmp/faiss_index.faiss'

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    pdf_url TEXT,
    page INT,
    text TEXT,
    b0 REAL, b1 REAL, b2 REAL, b3 REAL
)''')

for i, p in enumerate(paragraphs):
    cur.execute('INSERT INTO docs (pdf_url, page, text, b0, b1, b2, b3) VALUES (?,?,?,?,?,?,?)',
                (p['pdf_url'], p['page'], p['text'], *p['bbox']))
conn.commit()

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, index_path)

shutil.copy(db_path, pathlib.Path(DRIVE_DIR)/'docs.db')
shutil.copy(index_path, pathlib.Path(DRIVE_DIR)/'faiss_index.faiss')

```


```
print('PDF 개수:', len(pdf_items))
print('문단 개수:', len(paragraphs))
print('DB 경로:', str(pathlib.Path(DRIVE_DIR)/'docs.db'))
print('Faiss 인덱스 경로:', str(pathlib.Path(DRIVE_DIR)/'faiss_index.faiss'))

```
