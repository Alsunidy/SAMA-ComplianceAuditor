import re, json

IN_PATH = "../data/sama_csf/sama_raw.txt"
OUT_PATH = "controls.jsonl"

CONTROL_RE = re.compile(r'^\s*(\d+\.\d+\.\d+)\s+(.+?)\s*$')
CONTROL4_RE = re.compile(r'^\s*(\d+\.\d+\.\d+\.\d+)\s+(.+?)\s*$')
SUBDOMAIN_RE = re.compile(r'^\s*(\d+\.\d+)\s+([A-Z].+?)\s*$')
NOISE_RE = re.compile(r'^\s*(Version 1\.0.*Page \d+ of \d+|Page \d+ of \d+)\s*$')

with open(IN_PATH, encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

first_principle_idx = None
for i, line in enumerate(lines):
    if line.strip() == "Principle":
        first_principle_idx = i
        break

start_idx = 0
if first_principle_idx is not None:
    for i in range(first_principle_idx - 1, -1, -1):
        if CONTROL_RE.match(lines[i]):
            start_idx = i
            break
lines = lines[start_idx:]

for i, line in enumerate(lines):
    if line.strip() == "Appendices":
        lines = lines[:i]
        break

records = []
current = None
current_subdomain = "Cybersecurity Governance and Leadership"
mode = None

def flush():
    global current
    if current:
        current["principle"] = " ".join(current["principle"]).strip()
        current["objective"] = " ".join(current["objective"]).strip()
        current["considerations_text"] = "\n".join(current["considerations"]).strip()
        parent_line = f"Parent control: {current['parent_control_id']}\n" if current.get('parent_control_id') else ""
        text = (
            f"Control {current['control_id']} - {current['title']}\n"
            f"Domain: {current['domain']}\n"
            f"{parent_line}"
            f"Principle: {current['principle']}\n"
            f"Objective: {current['objective']}\n"
            f"Control considerations:\n{current['considerations_text']}"
        )
        current["text"] = text
        records.append(current)
        current = None

for raw_line in lines:
    line = raw_line.strip()

    if not line:
        continue
    if NOISE_RE.match(line):
        continue
    if line == "Version 1.0":
        continue

    m_control4 = CONTROL4_RE.match(raw_line)
    m_control = CONTROL_RE.match(raw_line)
    m_sub = SUBDOMAIN_RE.match(raw_line)

    if m_control4:
        flush()
        control_id, title = m_control4.group(1), m_control4.group(2)
        current = {
            "control_id": control_id,
            "title": title,
            "domain": current_subdomain,
            "parent_control_id": ".".join(control_id.split(".")[:3]),
            "principle": [],
            "objective": [],
            "considerations": [],
        }
        mode = None
        continue

    if m_control:
        flush()
        control_id, title = m_control.group(1), m_control.group(2)
        current = {
            "control_id": control_id,
            "title": title,
            "domain": current_subdomain,
            "parent_control_id": None,
            "principle": [],
            "objective": [],
            "considerations": [],
        }
        mode = None
        continue

    if m_sub and len(line.split()) < 12 and not line.endswith('.') and not line[0].islower():
        current_subdomain = m_sub.group(2)
        continue

    if current is None:
        continue

    if line == "Principle":
        mode = "principle"
        continue
    if line == "Objective":
        mode = "objective"
        continue
    if line.lower() == "control considerations":
        mode = "considerations"
        continue

    if mode == "principle":
        current["principle"].append(line)
    elif mode == "objective":
        current["objective"].append(line)
    elif mode == "considerations":
        current["considerations"].append(raw_line.rstrip())

flush()

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Parsed {len(records)} controls")
for r in records[:3]:
    print("---")
    print(r["control_id"], "|", r["title"], "|", r["domain"])
    print("Principle:", r["principle"][:100])
