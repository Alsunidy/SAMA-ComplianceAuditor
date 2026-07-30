import re, json

IN_PATH = "../data/sama_csf/sama_ar_raw.txt"
OUT_PATH = "controls_ar.jsonl"

# Bidi control chars pdftotext preserves around embedded LTR numbers in RTL text
BIDI = "‪‫‬‭‮"

def strip_bidi(s):
    return "".join(c for c in s if c not in BIDI)

# Heading pattern: optional bidi marks, spaces, digit groups separated by dots, then Arabic title
HEADING_RE = re.compile(r'^[\s‪‫‬\x0c]*(\d+(?:\.\d+){1,3})[‪‫‬]*([^\d].*)$')

def fix_control_id(raw_id):
    parts = raw_id.split(".")
    return ".".join(reversed(parts))

with open(IN_PATH, encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

# find first "المبدأ" (Principle) line to locate real content start
first_principle_idx = None
for i, line in enumerate(lines):
    if strip_bidi(line).strip() == "المبدأ":
        first_principle_idx = i
        break

start_idx = 0
if first_principle_idx is not None:
    for i in range(first_principle_idx - 1, -1, -1):
        m = HEADING_RE.match(lines[i])
        if m and len(m.group(1).split(".")) == 3:
            start_idx = i
            break
lines = lines[start_idx:]

# stop at appendices-equivalent (rough heuristic: look for "الملحق" section far down, else keep all)
for i, line in enumerate(lines):
    if strip_bidi(line).strip().startswith("الملحق"):
        lines = lines[:i]
        break

records = []
current = None
current_subdomain = "قيادة وحوكمة الأمن السيبراني"
mode = None

def flush():
    global current
    if current:
        current["principle"] = " ".join(current["principle"]).strip()
        current["objective"] = " ".join(current["objective"]).strip()
        current["considerations_text"] = "\n".join(current["considerations"]).strip()
        parent_line = f"الضابط الأصل: {current['parent_control_id']}\n" if current.get('parent_control_id') else ""
        text = (
            f"الضابط {current['control_id']} - {current['title']}\n"
            f"المجال: {current['domain']}\n"
            f"{parent_line}"
            f"المبدأ: {current['principle']}\n"
            f"الهدف: {current['objective']}\n"
            f"اعتبارات التحكم:\n{current['considerations_text']}"
        )
        current["text"] = text
        records.append(current)
        current = None

for raw_line in lines:
    clean = strip_bidi(raw_line).strip()
    if not clean:
        continue

    m = HEADING_RE.match(raw_line)
    if m:
        num_parts = m.group(1).split(".")
        title = strip_bidi(m.group(2)).strip()
        if len(num_parts) == 4:
            flush()
            control_id = fix_control_id(m.group(1))
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
        elif len(num_parts) == 3:
            flush()
            control_id = fix_control_id(m.group(1))
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
        elif len(num_parts) == 2 and title:
            current_subdomain = title
            continue

    if current is None:
        continue

    if clean == "المبدأ":
        mode = "principle"
        continue
    if clean == "الهدف":
        mode = "objective"
        continue
    if clean == "اعتبارات التحكم":
        mode = "considerations"
        continue
    if re.match(r'^\d+\.$', clean) or re.match(r'^\.\d+$', clean):
        # bullet number line like "1." on its own (Arabic list numbering artifact).
        # The extraction sometimes reverses this to ".1", so both orders are skipped.
        continue

    if mode == "principle":
        current["principle"].append(clean)
    elif mode == "objective":
        current["objective"].append(clean)
    elif mode == "considerations":
        current["considerations"].append(clean)

flush()

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Parsed {len(records)} controls (AR)")
for r in records[:5]:
    print(r["control_id"], "-", r["title"], "-", r["domain"])
