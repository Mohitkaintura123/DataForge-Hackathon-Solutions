# generate_csv.py

import json
import csv
from context_scoring import compute_relevance_scores

THRESHOLD = 0.15
PROJECT_LINK = "https://github.com/Mohitkaintura123/Self-Auditing-Context-Engine"

# Load documents from JSON
with open("sample_docs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

query = data["query"]
documents = data["documents"]

doc_ids = [doc["document_id"] for doc in documents]
doc_texts = [doc["summary"] for doc in documents]

# Debug check (VERY IMPORTANT)
print("Number of documents loaded:", len(documents))

# Compute relevance scores
scores = compute_relevance_scores(query, doc_texts)

rows = []

for i in range(len(documents)):
    score = round(scores[i], 2)

    if score >= THRESHOLD:
        decision = "Included"
        reason = "High semantic relevance to the user query"
    else:
        decision = "Excluded"
        reason = "Relevance score below defined threshold"

    rows.append([
        query,
        doc_ids[i],
        doc_texts[i],
        score,
        THRESHOLD,
        decision,
        reason,
        PROJECT_LINK
    ])

# Write CSV
output_path = "output/PS1_Context_Audit.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "query",
        "document_id",
        "document_summary",
        "relevance_score",
        "threshold",
        "decision",
        "reason",
        "project_link"
    ])
    writer.writerows(rows)

print("CSV generated at:", output_path)
