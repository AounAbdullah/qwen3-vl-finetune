"""
generate_cms1500.py — Day 2: synthetic CMS-1500 dataset generator.

JSON-FIRST DESIGN: ground-truth record is generated first, form image
is rendered FROM it. The label can never disagree with the image.

All values are synthetic. No PHI, ever.

Outputs (under --outdir):
  data/raw/form_XXXX.json            ground-truth record
  data/processed/form_XXXX.png       clean render
  data/processed/form_XXXX_scan.png  degraded "scanned" variant
  splits.json                        frozen train/val split (by RECORD)
  train.jsonl / val.jsonl            Qwen3-VL conversation format

Usage:
  python scripts/generate_cms1500.py --n 25 --seed 7 --outdir data
"""

import argparse
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# ---------------------------------------------------------------------------
# 1. VALUE POOLS
# ---------------------------------------------------------------------------
ICD10 = ["E11.9", "I10", "M54.5", "J06.9", "Z00.00", "K21.9", "F41.1", "M25.561"]
CPT   = ["99213", "99214", "99203", "80053", "85025", "71046", "36415", "97110"]
POS   = ["11", "21", "22", "23"]
PLANS = ["Aetna PPO", "BCBS Choice", "Cigna Open Access", "United HMO", "Humana Gold"]

_FIRST   = ["MARIA", "JAMES", "AISHA", "WEI", "CARLOS", "FATIMA", "JOHN", "PRIYA"]
_LAST    = ["GARCIA", "O'BRIEN-VELASQUEZ", "SMITH", "NGUYEN", "KHAN",
            "JOHNSON", "PATEL", "WASHINGTON-KOWALCZYK"]
_STREETS = ["MAPLE AVE", "2ND ST", "RIVERSIDE DR", "OAK LANE", "HILL RD"]
_CITIES  = [
    ("SPRINGFIELD", "IL", "62704"),
    ("AUSTIN",      "TX", "78701"),
    ("FRESNO",      "CA", "93701"),
    ("DAYTON",      "OH", "45402"),
]

try:
    from faker import Faker
    _fake = Faker()
    Faker.seed(0)

    def _name(rng):
        return f"{_fake.last_name().upper()}, {_fake.first_name().upper()}"

    def _street(rng):
        return _fake.street_address().upper()

except ImportError:
    def _name(rng):
        return f"{rng.choice(_LAST)}, {rng.choice(_FIRST)}"

    def _street(rng):
        return f"{rng.randint(10, 9999)} {rng.choice(_STREETS)}"


# ---------------------------------------------------------------------------
# 2. RECORD GENERATOR — ground truth first
# ---------------------------------------------------------------------------
def make_record(seed: int) -> dict:
    rng = random.Random(seed)
    city, state, zipc = rng.choice(_CITIES)

    # Skew realistic but force the 6-line edge case to exist
    n_lines = rng.choice([1, 1, 1, 2, 2, 3, 6])
    n_dx    = rng.randint(1, min(4, len(ICD10)))
    diagnoses = rng.sample(ICD10, n_dx)

    lines, total = [], 0.0
    for _ in range(n_lines):
        charge = round(rng.uniform(0, 450), 2)   # $0.00 charges are an edge case
        units  = rng.randint(1, 3)
        d      = f"2026-{rng.randint(1,5):02d}-{rng.randint(1,28):02d}"
        lines.append({
            "date_from":         d,
            "date_to":           d,
            "place_of_service":  rng.choice(POS),
            "cpt_code":          rng.choice(CPT),
            "modifier":          rng.choice([None, None, "25", "59"]),
            "diagnosis_pointer": str(rng.randint(1, n_dx)),
            "charges":           charge,
            "units":             units,
        })
        total += charge * units

    paid = round(rng.choice([0.0, 0.0, rng.uniform(0, total)]), 2)

    return {
        "patient": {
            "name":    _name(rng),
            "dob":     f"{rng.randint(1940,2008)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "sex":     rng.choice(["M", "F"]),
            "address": f"{_street(rng)}, {city}, {state} {zipc}",
        },
        "insured": {
            "id":        f"{rng.choice('ABCDXYZ')}{rng.randint(10**8, 10**9-1)}",
            "name":      _name(rng) if rng.random() < 0.3 else "SAME",
            "plan_name": rng.choice(PLANS),
        },
        "diagnoses":    diagnoses,
        "service_lines": lines,
        "provider": {
            "npi":    str(rng.randint(10**9, 10**10-1)),
            "tax_id": f"{rng.randint(10,99)}-{rng.randint(10**6, 10**7-1)}",
            "name":   f"DR. {_name(rng)}",
        },
        "totals": {
            "total_charge": round(total, 2),
            "amount_paid":  paid,
            "balance_due":  round(total - paid, 2),
        },
    }


# ---------------------------------------------------------------------------
# 3. RENDERER
# ---------------------------------------------------------------------------
W, H = 1700, 2200   # ~200 dpi US Letter


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def render_form(rec: dict, path: Path):
    img = Image.new("RGB", (W, H), "white")
    d   = ImageDraw.Draw(img)
    f_lab, f_val, f_hdr = _font(22), _font(30), _font(36, bold=True)

    def box(x, y, w, h, label, value=""):
        d.rectangle([x, y, x+w, y+h], outline="black", width=2)
        d.text((x+8,  y+6),  label, font=f_lab, fill="black")
        if value:
            d.text((x+12, y+36), str(value), font=f_val, fill="black")

    d.text((60, 40), "HEALTH INSURANCE CLAIM FORM", font=f_hdr, fill="black")
    d.text((60, 90), "SYNTHETIC TRAINING DATA — NOT A REAL CLAIM (CMS-1500 style)",
           font=f_lab, fill="black")

    p, ins, prov, tot = rec["patient"], rec["insured"], rec["provider"], rec["totals"]

    box(60,   140, 780, 90, "2. PATIENT'S NAME (Last, First)", p["name"])
    box(860,  140, 360, 90, "3. PATIENT'S BIRTH DATE",         p["dob"])
    box(1240, 140, 180, 90, "SEX",                             p["sex"])
    box(60,   250, 780, 90, "5. PATIENT'S ADDRESS",            p["address"])
    box(860,  250, 560, 90, "1a. INSURED'S I.D. NUMBER",       ins["id"])
    box(60,   360, 780, 90, "4. INSURED'S NAME",               ins["name"])
    box(860,  360, 560, 90, "11c. INSURANCE PLAN NAME",        ins["plan_name"])
    box(60,   470, 1360, 90, "21. DIAGNOSIS CODES (ICD-10)",   "   ".join(rec["diagnoses"]))

    # Box 24: always draw 6 rows — empty rows are an edge case the model must handle
    ty = 600
    d.text((60, ty), "24. SERVICE LINES", font=f_lab, fill="black")
    cols = [
        (60,   "FROM"),  (260,  "TO"),      (460,  "POS"),
        (560,  "CPT"),   (720,  "MOD"),     (840,  "DX PTR"),
        (980,  "CHARGES"), (1200, "UNITS"),
    ]
    hdr_y = ty + 34
    for cx, cname in cols:
        d.text((cx+6, hdr_y+6), cname, font=f_lab, fill="black")

    for r in range(6):
        ry = hdr_y + 44 + r * 64
        d.rectangle([60, ry, 1420, ry+64], outline="black", width=1)
        if r < len(rec["service_lines"]):
            ln   = rec["service_lines"][r]
            vals = [
                ln["date_from"], ln["date_to"], ln["place_of_service"],
                ln["cpt_code"],  ln["modifier"] or "", ln["diagnosis_pointer"],
                f"{ln['charges']:.2f}", str(ln["units"]),
            ]
            for (cx, _), v in zip(cols, vals):
                d.text((cx+6, ry+16), str(v), font=f_val, fill="black")

    by = hdr_y + 44 + 6*64 + 40
    box(60,  by,      440, 90, "25. FEDERAL TAX I.D.",    prov["tax_id"])
    box(520, by,      440, 90, "33a. NPI",                prov["npi"])
    box(980, by,      440, 90, "33. BILLING PROVIDER",    prov["name"])
    box(60,  by+110,  440, 90, "28. TOTAL CHARGE",  f"$ {tot['total_charge']:.2f}")
    box(520, by+110,  440, 90, "29. AMOUNT PAID",   f"$ {tot['amount_paid']:.2f}")
    box(980, by+110,  440, 90, "30. BALANCE DUE",   f"$ {tot['balance_due']:.2f}")

    img.save(path)
    return img


# ---------------------------------------------------------------------------
# 4. DEGRADATION — clean render → "scanned" variant (same record!)
# ---------------------------------------------------------------------------
def degrade(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    out = img.rotate(rng.uniform(-1.0, 1.0), expand=False, fillcolor="white")
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.88, 1.08))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.0))
    noise = Image.effect_noise(out.size, rng.uniform(8, 18)).convert("L")
    out   = Image.composite(
        out,
        Image.new("RGB", out.size, "gray"),
        noise.point(lambda v: 255 if v > 30 else 200),
    )
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(35, 55))
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# 5. QWEN3-VL CONVERSATION FORMAT
# ---------------------------------------------------------------------------
SYSTEM      = "You are a medical claim form parser. Output only valid JSON."
INSTRUCTION = (
    "Extract all fields from this CMS-1500 form as JSON with keys: "
    "patient, insured, diagnoses, service_lines, provider, totals. "
    "Use null for empty fields and YYYY-MM-DD for all dates."
)


def to_messages(image_path: str, rec: dict) -> dict:
    target = json.dumps(rec, sort_keys=True)   # sort_keys → consistent key order every time
    return {"messages": [
        {"role": "system",    "content": [{"type": "text",  "text": SYSTEM}]},
        {"role": "user",      "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": INSTRUCTION},
        ]},
        {"role": "assistant", "content": [{"type": "text",  "text": target}]},
    ]}


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,  default=25)
    ap.add_argument("--seed",   type=int,  default=7)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw  = args.outdir / "raw"
    proc = args.outdir / "processed"
    raw.mkdir(parents=True,  exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    examples = {}   # record_id → list of (image_path, record)

    for i in range(args.n):
        rec = make_record(args.seed * 10_000 + i)
        rid = f"form_{i:04d}"

        # Ground truth JSON
        (raw / f"{rid}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))

        # Clean render
        clean_p = proc / f"{rid}.png"
        img     = render_form(rec, clean_p)

        # Degraded "scan" variant — SAME record, different looking image
        scan_p = proc / f"{rid}_scan.png"
        degrade(img, seed=args.seed * 10_000 + i).save(scan_p)

        examples[rid] = [(str(clean_p), rec), (str(scan_p), rec)]
        print(f"  generated {rid}")

    # Split BY RECORD so both variants of a form land on the same side
    ids = sorted(examples)
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, round(0.2 * len(ids)))
    split = {"val": sorted(ids[:n_val]), "train": sorted(ids[n_val:])}
    (args.outdir / "splits.json").write_text(json.dumps(split, indent=2))

    # Write JSONL files
    for part in ("train", "val"):
        with open(args.outdir / f"{part}.jsonl", "w") as f:
            for rid in split[part]:
                for img_path, rec in examples[rid]:
                    f.write(json.dumps(to_messages(img_path, rec)) + "\n")

    n_tr = sum(len(examples[r]) for r in split["train"])
    n_va = sum(len(examples[r]) for r in split["val"])
    print(f"\n✅ Done!")
    print(f"   Records  : {len(ids)}  (train: {len(split['train'])}  val: {len(split['val'])})")
    print(f"   Examples : train={n_tr}  val={n_va}  (2x records because clean+scan)")
    print(f"   Files    : {args.outdir}/splits.json, train.jsonl, val.jsonl")


if __name__ == "__main__":
    main()