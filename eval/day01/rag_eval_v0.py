import json, csv, time

def recall_at_k(relevant, retrieved, k):
    return int(any(doc in relevant for doc in retrieved[:k]))

def mrr(relevant, retrieved):
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

with open("day01/gold_qa.json") as f:
    gold_set = json.load(f)

recalls5, recalls10, mrrs = [], [], []
latencies = []

for item in gold_set:
    start = time.time()
    retrieved = ["d1","dX","dY"]
    relevant = item["gold_doc_ids"]
    recalls5.append(recall_at_k(relevant, retrieved, 5))
    recalls10.append(recall_at_k(relevant, retrieved, 10))
    mrrs.append(mrr(relevant, retrieved))
    latencies.append((time.time() - start)*1000)

row = [
    time.strftime("%Y-%m-%d-%H%M%S"),
    "KEYWORD_OVERLAP",
    '{"type":"keyword"}',
    10,
    sum(recalls5)/len(recalls5),
    sum(recalls10)/len(recalls10),
    sum(mrrs)/len(mrrs),
    sorted(latencies)[len(latencies)//2],
    max(latencies),
    "Day-1 toy baseline"
]

with open("day01/runs.csv","a",newline="") as f:
    csv.writer(f).writerow(row)

print("Metrics: Recall@5, Recall@10, MRR:", row[4], row[5], row[6])
