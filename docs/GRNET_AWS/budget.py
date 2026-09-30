"""Budget model for the GRNET-AWS request: measured timings -> work plan -> credits.

Single source of truth for every number in proposal.tex. Inputs (all in data/):
  measured_timings.json  per-phase wall-clock parsed from our A100 training logs (tools/extract_timings.py)
  bench_results.jsonl    ViT-B/L/g throughput on 1x A100-80GB, same code path (tools/bench_scaling.py)
  aws_prices.json        public AWS on-demand prices + ECB USD/EUR rate (tools/fetch_aws_prices.py)
Outputs: generated/*.tex (macros + tables, \\input by proposal.tex) and figs/*.pdf.

    python docs/GRNET_AWS/budget.py        # from the repository root
"""
import json
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)

# --------------------------------------------------------------------------- inputs
M = json.load(open(D("data", "measured_timings.json")))
P = json.load(open(D("data", "aws_prices.json")))
BENCH = [json.loads(line) for line in open(D("data", "bench_results.jsonl")) if line.strip()]

# --------------------------------------------------------------------------- policy
SLACK = 1.25          # every measured job duration is budgeted +25 %
PACKING = 0.90        # the node bills 8 GPUs; assume 10 % of GPU-slots idle while it is up
CONTINGENCY = 0.20    # failed/repeated runs, debugging, extra seeds, price and exchange-rate drift (GPU node only)
GPUS = 8              # p4de.24xlarge = 8x NVIDIA A100-SXM4-80GB
MONTHS = 6
REGION, NODE = "us-east-1", "p4de.24xlarge"

USD_PER_EUR = P["usd_per_eur"]
NODE_USD_H = P["ec2_usd_per_h"][NODE][REGION]

# --------------------------------------------------------------------------- measured unit costs
e224 = M["ssl_vitl_224_epoch_h"]["mean"]          # ViT-L SSL epoch, 224 px bulk (191,170 images)
e518 = M["ssl_vitl_518_epoch_h"]["mean"]          # ViT-L SSL epoch, 518 px
p2 = M["phase2_unfreeze4"]["run_hours_mean"]      # Phase-2 run, last 4 blocks unfrozen
fullft = M["phase2_fullft"]["run_hours_mean"]     # Phase-2 run, all 24 blocks unfrozen
probe = M["probe_frozen"]["run_hours_mean"]       # frozen-encoder probe (<= 25 epochs)
pred = M["predict"]
single = [v for v in pred.values() if v["members"] == 1]
sec_per_img_1m = sum(v["minutes"] * 60 / v["images"] for v in single) / len(single)   # 1 model x 9 TTA views
test5 = next(v["minutes"] for v in pred.values() if v["members"] == 5) / 60          # 5 models x 9 views, 619 imgs
N_LABELED = 6743
oof = N_LABELED * sec_per_img_1m / 3600          # out-of-fold pass of a 5-fold ensemble (1 member / image)
mpe4, mpe24 = M["phase2_unfreeze4"]["min_per_epoch_median"], M["phase2_fullft"]["min_per_epoch_median"]
r_unfreeze = lambda k: (mpe4 + (mpe24 - mpe4) * (k - 4) / 20) / mpe4   # linear in #unfrozen blocks


# --------------------------------------------------------------------------- capacity scaling (benchmark)
def bench(kind, bb, res, unfreeze=None):
    for b in BENCH:
        if (b["kind"] == kind and b["backbone"] == f"dinov2_vit{bb}14_reg" and b["res"] == res and "img_s" in b
                and (unfreeze is None or b.get("unfreeze") == unfreeze)):
            return b
    raise KeyError((kind, bb, res, unfreeze))


def ratio(kind, bb, res, unfreeze=None):  # wall-clock of size bb relative to ViT-L
    return bench(kind, "l", res, unfreeze)["img_s"] / bench(kind, bb, res, unfreeze)["img_s"]


R = {bb: dict(s224=ratio("ssl", bb, 224), s518=ratio("ssl", bb, 518), p2=ratio("phase2", bb, 518, 4),
              probe=ratio("phase2", bb, 518, 0)) for bb in ("b", "g")}


# --------------------------------------------------------------------------- work plan
class Job:
    def __init__(self, wp, label, cat, n, unit, months):
        self.wp, self.label, self.cat, self.n, self.unit, self.months = wp, label, cat, n, unit, months

    measured = property(lambda s: s.n * s.unit)
    budget = property(lambda s: s.n * s.unit * SLACK)


J = []
add = lambda *a: J.append(Job(*a))
# WP0 -- set-up and timing parity on AWS (the H100 pilot is a separate p5.4xlarge line)
add("WP0", "Parity: 1 SSL epoch at 224 px + 1 at 518 px (ViT-L)", "ssl", 1, e224 + e518, [1])
add("WP0", "Parity: 1 Phase-2 run + 1 frozen probe (fold 0)", "sup", 1, p2 + probe, [1])
# WP1 -- leakage-free re-baselining (Q1)
add("WP1", "Phase-2, 6 reference configurations x 5 loop-grouped folds", "sup", 30, p2, [1, 2])
add("WP1", "Frozen probes, 12 existing encoders x 5 folds", "sup", 60, probe, [1, 2])
add("WP1", "Out-of-fold inference, 9-view TTA, 6 ensembles", "sup", 6, oof, [2])
# WP2 -- SSL schedule, resolution and data (Q2), ViT-L
add("WP2", "SSL bulk, 60 ep at 224 px, 2 new seeds", "ssl", 2, 60 * e224, [2, 3])
add("WP2", "518-px tails (5 ep) from ep20 and ep60, 2 new seeds", "ssl", 4, 5 * e518, [3])
add("WP2", "Tail sweep, seed 42: ep20/ep40 + 5 ep; ep20 + 2/10 ep", "ssl", 4, 22 / 4 * e518, [2, 3])
add("WP2", "Full-resolution control, 518 px, ep31 -> ep60", "ssl", 1, 30 * e518, [2, 3])
add("WP2", "Unlabeled-data scaling: 10/25/50 % subsets (20 + 5 ep)", "ssl", 3, 20 * e224 + 5 * e518, [3])
add("WP2", "Frozen probes, 22 new checkpoints x 5 folds", "sup", 110, probe, [3, 4])
add("WP2", "Phase-2, top-3 SSL checkpoints x 5 folds", "sup", 15, p2, [4, 5])
# WP3 -- encoder capacity: ViT-B/14 and ViT-g/14 (Q3); ViT-L comes from WP1-WP2
for bb, name in (("b", "ViT-B"), ("g", "ViT-g")):
    add("WP3", f"{name}: SSL bulk, 60 ep at 224 px", "ssl", 1, 60 * e224 * R[bb]["s224"], [3, 4])
    add("WP3", f"{name}: 518-px tails (5 ep) from ep20 and ep60", "ssl", 2, 5 * e518 * R[bb]["s518"], [3, 4])
    add("WP3", f"{name}: frozen probes, 3 encoders x 5 folds", "sup", 15, probe * R[bb]["probe"], [4])
    add("WP3", f"{name}: Phase-2, no-SSL vs SSL x 5 folds", "sup", 10, p2 * R[bb]["p2"], [4, 5])
# WP4 -- Phase-2 design and metric-aligned objectives (Q4), ViT-L
add("WP4", "Fine-tuning depth: 8 blocks x 5 folds", "sup", 5, p2 * r_unfreeze(8), [4])
add("WP4", "Fine-tuning depth: 12 blocks x 5 folds", "sup", 5, p2 * r_unfreeze(12), [4])
add("WP4", "Fine-tuning depth: all 24 blocks x 5 folds", "sup", 5, fullft, [5])
add("WP4", "Multi-level taps (2 variants) x 5 folds", "sup", 10, p2, [4, 5])
add("WP4", "Losses: measurement-aware (2 weights), Wing, smooth-L1 x 5 folds", "sup", 20, p2, [5])
add("WP4", "Sampler temperature (0, 1) + strong augmentation x 5 folds", "sup", 15, p2, [5])
add("WP4", "Best configuration, 2 extra seeds x 5 folds", "sup", 10, p2, [5, 6])
add("WP4", "Out-of-fold inference, 10 ensembles", "sup", 10, oof, [5])
# WP5 -- final models, ensembles, uncertainty, release
add("WP5", "ViT-L final: 5 folds x 3 seeds + all-data model", "sup", 16, p2, [6])
add("WP5", "ViT-g final: 5 folds + all-data model", "sup", 6, p2 * R["g"]["p2"], [6])
add("WP5", "OOF + test inference, multi-scale/intensity TTA (4 ensembles)", "sup", 1,
    3 * oof + oof * R["g"]["probe"] + 4 * test5, [6])

WPS = OrderedDict([
    ("WP0", "Set-up, timing parity, H100 pilot"),
    ("WP1", "Leakage-free re-baselining (Q1)"),
    ("WP2", "SSL schedule, resolution and data (Q2)"),
    ("WP3", "Encoder capacity ViT-B/L/g (Q3)"),
    ("WP4", "Phase-2 design and objectives (Q4)"),
    ("WP5", "Final ensembles, uncertainty, release"),
])
SPANS = {"WP0": (0.0, 0.5), "WP1": (0.25, 2.0), "WP2": (1.0, 4.0), "WP3": (2.0, 5.0), "WP4": (3.0, 5.0),
         "WP5": (4.0, 6.0)}

# --------------------------------------------------------------------------- aggregate
node_h_of = lambda gpu_h: gpu_h / (GPUS * PACKING) * (1 + CONTINGENCY)
meas = sum(j.measured for j in J)
budg = sum(j.budget for j in J)
packed = budg / PACKING
node_h_exact = node_h_of(budg)
node_h = math.ceil(node_h_exact / MONTHS) * MONTHS   # whole instance-hours per month
gpu_h_billed = node_h * GPUS

by_wp = OrderedDict((wp, [j for j in J if j.wp == wp]) for wp in WPS)
month_gpu = defaultdict(lambda: defaultdict(float))
for j in J:
    for m in j.months:
        month_gpu[m][j.cat] += j.budget / len(j.months)

# other services (monthly quantities, us-east-1)
EBS_GB, S3_GB, EGRESS_GB_TOTAL, T3_H_MONTH, PILOT_H = 1000, 1000, 300, 100, 12
S3_PUT_MONTH, S3_GET_MONTH = 100_000, 1_000_000
S3_REQ_USD_MONTH = S3_PUT_MONTH * 0.005 / 1000 + S3_GET_MONTH * 0.0004 / 1000   # S3 Standard request prices
t3 = P["ec2_usd_per_h"]["t3.large"][REGION]
p5 = P["ec2_usd_per_h"]["p5.4xlarge"][REGION]
services = [
    # (service, configuration, quantity (6 months), unit, unit price, usd)
    ("Amazon EC2", f"{NODE}, On-Demand Linux (8x A100-80GB)", node_h, "instance-h", NODE_USD_H, node_h * NODE_USD_H),
    ("Amazon EC2", "p5.4xlarge, On-Demand Linux (1x H100-80GB), WP0 pilot", PILOT_H, "instance-h", p5, PILOT_H * p5),
    ("Amazon EC2", "t3.large, On-Demand Linux (control/staging node)", T3_H_MONTH * MONTHS, "instance-h", t3,
     T3_H_MONTH * MONTHS * t3),
    ("Amazon EBS", f"gp3, {EBS_GB:,} GB (3,000 IOPS, 125 MB/s baseline)", EBS_GB * MONTHS, "GB-month",
     P["ebs_gp3_usd_gb_month"], EBS_GB * MONTHS * P["ebs_gp3_usd_gb_month"]),
    ("Amazon S3", f"Standard, {S3_GB:,} GB average (data, checkpoints, logs, container image)", S3_GB * MONTHS, "GB-month",
     P["s3_standard_usd_gb_month"], S3_GB * MONTHS * P["s3_standard_usd_gb_month"]),
    ("Amazon S3", "Requests: 100k PUT/LIST + 1M GET per month", MONTHS, "month", S3_REQ_USD_MONTH,
     MONTHS * S3_REQ_USD_MONTH),
    ("Data transfer", "Internet egress (models, predictions, logs)", EGRESS_GB_TOTAL, "GB", P["egress_usd_gb"],
     EGRESS_GB_TOTAL * P["egress_usd_gb"]),
]
total_usd = sum(s[-1] for s in services)
total_eur = total_usd / USD_PER_EUR
request_eur = round(total_eur)   # request = calculator estimate, in whole euros
compute_usd = services[0][-1] + services[1][-1] + services[2][-1]
node_usd = services[0][-1]
eu_node = P["ec2_usd_per_h"][NODE]["eu-central-1"]

# monthly burn (the p5 pilot sits in M1; fixed monthly costs spread evenly)
fixed_month = sum(s[-1] for s in services[2:]) / MONTHS
month_rows = []
for m in range(1, MONTHS + 1):
    g = sum(month_gpu[m].values())
    nh = node_h * g / budg          # the rounded node-hours, apportioned by monthly GPU use
    usd = nh * NODE_USD_H + fixed_month + (services[1][-1] if m == 1 else 0.0)
    month_rows.append((m, month_gpu[m]["ssl"], month_gpu[m]["sup"], g, nh, usd, usd / USD_PER_EUR))

longest = max(J, key=lambda j: j.unit)   # longest single job (critical path inside a burst)
usd_per_budget_h = node_usd / budg        # effective $ per budgeted A100-hour (packing + contingency + rounding)
wp_usd = {wp: sum(j.budget for j in jobs) * usd_per_budget_h for wp, jobs in by_wp.items()}
mvs_jobs = [j for j in J if j.wp in ("WP0", "WP1", "WP2") or (j.wp == "WP5" and "ViT-g" not in j.label)]
mvs_node_h = math.ceil(node_h_of(sum(j.budget for j in mvs_jobs)) / MONTHS) * MONTHS
mvs_usd = mvs_node_h * NODE_USD_H + sum(x[-1] for x in services[1:])


# --------------------------------------------------------------------------- LaTeX output
def f0(x): return f"{x:,.0f}"
def f1(x): return f"{x:,.1f}"
def f2(x): return f"{x:,.2f}"


macros = {
    "SlackPct": f0((SLACK - 1) * 100), "PackingPct": f0(PACKING * 100), "ContPct": f0(CONTINGENCY * 100),
    "NodeUSDh": f"{NODE_USD_H:.5f}".rstrip("0"), "NodeUSDhShort": f2(NODE_USD_H), "GPUUSDh": f2(NODE_USD_H / GPUS),
    "EUNodeUSDh": f2(eu_node), "EUPremiumPct": f1((eu_node / NODE_USD_H - 1) * 100),
    "EUExtraUSD": f0(node_h * (eu_node - NODE_USD_H)),
    "PfiveUSDh": f2(p5), "PfourdUSDh": f2(P["ec2_usd_per_h"]["p4d.24xlarge"][REGION]),
    "PfourdGPUUSDh": f2(P["ec2_usd_per_h"]["p4d.24xlarge"][REGION] / GPUS),
    "HBreakEven": f1(p5 / (NODE_USD_H / GPUS)),
    "FX": f"{USD_PER_EUR:.4f}", "FXdate": P["usd_per_eur_date"], "PriceDate": P["ec2_published"][:10],
    "GPUhMeasured": f0(meas), "GPUhBudget": f0(budg), "GPUhPacked": f0(packed), "GPUhBilled": f0(gpu_h_billed),
    "NodeH": f0(node_h), "NodeDays": f1(node_h / 24), "NodeHMonth": f1(node_h / MONTHS),
    "NodeUSD": f0(node_usd), "ComputeUSD": f0(compute_usd), "TotalUSD": f0(total_usd), "TotalEUR": f0(total_eur),
    "RequestEUR": f0(request_eur), "MonthlyUSD": f0(total_usd / MONTHS), "MonthlyEUR": f0(total_eur / MONTHS),
    "NodeSharePct": f0(node_usd / total_usd * 100), "NRuns": f0(sum(j.n for j in J)),
    "OneGPUDays": f0(budg / 24), "TwoGPUDays": f0(budg / 2 / 24),
    # measured unit costs
    "EpochTwoTwoFour": f2(e224), "EpochTwoTwoFourMin": f0(e224 * 60), "EpochFiveOneEight": f2(e518),
    "ImgsTwoTwoFour": f1(M["ssl_vitl_224_epoch_h"]["img_per_s"]), "ImgsFiveOneEight": f1(M["ssl_vitl_518_epoch_h"]["img_per_s"]),
    "ResRatio": f1(e518 / e224), "PaperRecipeH": f1(M["ssl_paper_recipe_h"]), "FullResH": f0(100 * e518),
    "PtwoRunH": f1(p2), "PtwoRuns": str(M["phase2_unfreeze4"]["n"]), "PtwoMaxH": f1(M["phase2_unfreeze4"]["run_hours_max"]),
    "PtwoEpochs": f0(M["phase2_unfreeze4"]["epochs_mean"]), "PtwoMinEpoch": f1(mpe4),
    "FullFTH": f1(fullft), "FullFTMinEpoch": f1(mpe24), "ProbeH": f1(probe), "ProbeRuns": str(M["probe_frozen"]["n"]),
    "OOFH": f1(oof), "TestFiveH": f2(test5), "SecPerImg": f2(sec_per_img_1m),
    # capacity ratios
    "RgSSL": f1(R["g"]["s224"]), "RgSSLhi": f1(R["g"]["s518"]), "RgPtwo": f1(R["g"]["p2"]), "RgProbe": f1(R["g"]["probe"]),
    "RbSSL": f2(R["b"]["s224"]), "RbPtwo": f2(R["b"]["p2"]),
    "ParamRatio": f1(bench("ssl", "g", 224)["params_M"] / bench("ssl", "l", 224)["params_M"]),
    "TokenRatio": f1((518 // 14) ** 2 / (224 // 14) ** 2),
    "BenchAgreePct": f0(math.ceil(100 * max(abs(bench("ssl", "l", r)["img_s"] / M[f"ssl_vitl_{r}_epoch_h"]["img_per_s"] - 1)
                                            for r in (224, 518)))),
    "LongestJob": longest.label.replace("->", r"$\rightarrow$"), "LongestJobH": f0(longest.unit * SLACK),
    "LongestJobDays": f1(longest.unit * SLACK / 24),
}
for wp, jobs in by_wp.items():
    macros[f"GPUh{wp}"] = f0(sum(j.budget for j in jobs))
    macros[f"USD{wp}"] = f0(wp_usd[wp])
    macros[f"EUR{wp}"] = f0(wp_usd[wp] / USD_PER_EUR)
macros.update({
    "USDperAh": f2(usd_per_budget_h), "CostPtwoRun": f0(p2 * SLACK * usd_per_budget_h),
    "CostProbe": f0(probe * SLACK * usd_per_budget_h), "CostPaperSSL": f0(M["ssl_paper_recipe_h"] * SLACK * usd_per_budget_h),
    "CostViTgSSL": f0(sum(j.budget for j in J if j.wp == "WP3" and "ViT-g" in j.label and j.cat == "ssl") * usd_per_budget_h),
    "MVSNodeH": f0(mvs_node_h), "MVSUSD": f0(mvs_usd), "MVSEUR": f0(mvs_usd / USD_PER_EUR),
    "MVSPct": f0(mvs_usd / total_usd * 100),
})
os.makedirs(D("generated"), exist_ok=True)
with open(D("generated", "numbers.tex"), "w") as fh:
    fh.write("% generated by budget.py -- do not edit\n")
    for k, v in macros.items():
        name = k.replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three") \
                .replace("4", "Four").replace("5", "Five")
        fh.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")

esc = lambda s: s.replace("%", r"\%").replace("->", r"$\rightarrow$").replace("~", r"$\sim$").replace("_", r"\_")
esc_x = lambda s: esc(s).replace(" x ", r" $\times$ ")

# Table: unit costs (measured -> budgeted)
rows = [
    ("SSL epoch, 224 px bulk (191,170 images)", f"{e224:.2f}", "100 epochs, phase1\\_dinov2", e224),
    ("SSL epoch, 518 px (tail / full-resolution)", f"{e518:.2f}", f"{M['ssl_vitl_518_epoch_h']['n']} epochs, 3 runs", e518),
    ("Paper SSL recipe (100 ep @224 + 4 ep @518)", f"{M['ssl_paper_recipe_h']:.1f}", "start to finish", M["ssl_paper_recipe_h"]),
    ("Phase-2 run, 4 blocks unfrozen", f"{p2:.2f}", f"mean of {M['phase2_unfreeze4']['n']} runs, "
     f"{M['phase2_unfreeze4']['epochs_min']}--{M['phase2_unfreeze4']['epochs_max']} ep", p2),
    ("Phase-2 run, all 24 blocks unfrozen", f"{fullft:.2f}", f"mean of {M['phase2_fullft']['n']} runs", fullft),
    ("Frozen-encoder probe, $\\le$25 ep", f"{probe:.2f}", f"mean of {M['probe_frozen']['n']} runs", probe),
    ("OOF inference, 6,743 images, 9 TTA views", f"{oof:.2f}", f"{sec_per_img_1m:.2f} s/image measured", oof),
    ("Test inference, 5 models $\\times$ 9 views, 619 images", f"{test5:.2f}", "measured", test5),
]
with open(D("generated", "tab_unit.tex"), "w") as fh:
    for name, m_, src, v in rows:
        bud = f"{v * SLACK:.2f}" if v < 50 else f"{v * SLACK:.1f}"
        fh.write(f"{name} & {m_} & {bud} & \\${v * SLACK * usd_per_budget_h:,.0f} & {src} \\\\\n")

# Table: capacity scaling (production vs benchmark)
P2_EPOCH_IMGS = 450 * 64   # sampler-defined Phase-2 epoch: 450 task-homogeneous batches of 64
prod = {("ssl", 224, None): M["ssl_vitl_224_epoch_h"]["img_per_s"], ("ssl", 518, None): M["ssl_vitl_518_epoch_h"]["img_per_s"],
        ("phase2", 518, 4): P2_EPOCH_IMGS / (mpe4 * 60), ("phase2", 518, 0): P2_EPOCH_IMGS / (M["probe_frozen"]["min_per_epoch_median"] * 60)}
with open(D("generated", "tab_scaling.tex"), "w") as fh:
    for kind, res, unf, label in (("ssl", 224, None, "SSL, 224\\,px"), ("ssl", 518, None, "SSL, 518\\,px"),
                                  ("phase2", 518, 4, "Fine-tuning"), ("phase2", 518, 0, "Frozen probe")):
        cells = []
        for bb in ("b", "l", "g"):
            b = bench(kind, bb, res, unf)
            mb = f", mb\\,{b['bs']}" if kind == "ssl" and bb == "g" else ""
            cells.append(f"{b['img_s']:.1f} ({b['peak_gb']:.0f}\\,GB{mb})")
        pcell = f"{prod[(kind, res, unf)]:.1f}"
        ratio_g = bench(kind, "l", res, unf)["img_s"] / bench(kind, "g", res, unf)["img_s"]
        fh.write(f"{label} & {cells[0]} & {cells[1]} & {pcell} & {cells[2]} & {ratio_g:.2f}$\\times$ \\\\\n")

# Table: work packages
with open(D("generated", "tab_wp.tex"), "w") as fh:
    for wp, jobs in by_wp.items():
        fh.write(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{wp}}} -- {esc(WPS[wp])}}} \\\\\n")
        for j in jobs:
            fh.write(f"\\quad {esc_x(j.label)} & {j.n} & {j.unit:,.1f} & {j.measured:,.0f} & {j.budget:,.0f} \\\\\n")
        fh.write(f"\\addlinespace[1pt]\\multicolumn{{3}}{{r}}{{\\emph{{{wp} subtotal}}}} & "
                 f"{sum(j.measured for j in jobs):,.0f} & \\textbf{{{sum(j.budget for j in jobs):,.0f}}} \\\\\\addlinespace[2pt]\n")

# Table: from measured A100-hours to requested credits
with open(D("generated", "tab_waterfall.tex"), "w") as fh:
    fh.write(f"Measured-rate A100-hours of the work plan & {meas:,.0f} & -- \\\\\n")
    fh.write(f"+{(SLACK - 1) * 100:.0f}\\,\\% slack on every job & {budg:,.0f} & -- \\\\\n")
    fh.write(f"$\\div$\\,{PACKING:.2f} packing (8 GPUs billed per node) & {packed:,.0f} & {packed / GPUS:,.0f} \\\\\n")
    fh.write(f"+{CONTINGENCY * 100:.0f}\\,\\% contingency, rounded up to whole hours/month & {gpu_h_billed:,.0f} & "
             f"\\textbf{{{node_h:,.0f}}} \\\\\n")

# Table: services (= AWS Pricing Calculator lines)
with open(D("generated", "tab_services.tex"), "w") as fh:
    for svc, conf, q, unit, up, usd in services:
        upf = f"{up:.5f}".rstrip("0").rstrip(".") if up < 1 else f"{up:,.5f}".rstrip("0").rstrip(".")
        fh.write(f"{svc} & {esc(conf)} & {q:,.0f} {unit} & \\${upf} & \\${usd:,.2f} & \\EUR{{{usd / USD_PER_EUR:,.2f}}} \\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"\\multicolumn{{4}}{{l}}{{\\textbf{{Total, 6 months}} (\\${total_usd / MONTHS:,.2f} per month)}} & "
             f"\\textbf{{\\${total_usd:,.2f}}} & \\textbf{{\\EUR{{{total_eur:,.2f}}}}} \\\\\n")

# Table: monthly burn
with open(D("generated", "tab_months.tex"), "w") as fh:
    acts = {1: "WP0 set-up and parity; WP1 starts", 2: "WP1 completes; WP2 SSL runs start",
            3: "WP2 SSL + probes; WP3 SSL (ViT-B, ViT-g)", 4: "WP2 Phase-2; WP3 probes/Phase-2; WP4 starts",
            5: "WP3 and WP4 complete; WP5 starts", 6: "WP5 final ensembles, inference, release, report"}
    for m, ssl, sup, g, nh, usd, eur in month_rows:
        fh.write(f"M{m} & {acts[m]} & {ssl:,.0f} & {sup:,.0f} & {nh:,.0f} & \\${usd:,.0f} & \\EUR{{{eur:,.0f}}} \\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"\\multicolumn{{2}}{{l}}{{\\textbf{{Total}}}} & {sum(r[1] for r in month_rows):,.0f} & "
             f"{sum(r[2] for r in month_rows):,.0f} & {sum(r[4] for r in month_rows):,.0f} & "
             f"\\textbf{{\\${sum(r[5] for r in month_rows):,.0f}}} & \\textbf{{\\EUR{{{sum(r[6] for r in month_rows):,.0f}}}}} \\\\\n")

# Calculator inputs (monthly averages; 6 x monthly = request)
with open(D("generated", "tab_calculator.tex"), "w") as fh:
    fh.write(f"Amazon EC2 & {REGION} & {NODE}, Linux, On-Demand, 1 instance, "
             f"{node_h / MONTHS:,.0f} h/month & \\${node_usd / MONTHS:,.2f} \\\\\n")
    fh.write(f"Amazon EC2 & {REGION} & p5.4xlarge, Linux, On-Demand, 1 instance, {PILOT_H / MONTHS:.0f} h/month "
             f"(the {PILOT_H}-h pilot) & \\${PILOT_H * p5 / MONTHS:,.2f} \\\\\n")
    fh.write(f"Amazon EC2 & {REGION} & t3.large, Linux, On-Demand, 1 instance, {T3_H_MONTH} h/month & "
             f"\\${T3_H_MONTH * t3:,.2f} \\\\\n")
    fh.write(f"Amazon EBS & {REGION} & 1 gp3 volume, {EBS_GB:,} GB, 3,000 IOPS, 125 MB/s, 730 h/month, no snapshots & "
             f"\\${EBS_GB * P['ebs_gp3_usd_gb_month']:,.2f} \\\\\n")
    fh.write(f"Amazon S3 & {REGION} & S3 Standard {S3_GB:,} GB-month; {S3_PUT_MONTH:,} PUT/LIST, {S3_GET_MONTH:,} GET & "
             f"\\${S3_GB * P['s3_standard_usd_gb_month'] + S3_REQ_USD_MONTH:,.2f} \\\\\n")
    fh.write(f"Data transfer & {REGION} & Outbound to internet {EGRESS_GB_TOTAL / MONTHS:.0f} GB/month & "
             f"\\${EGRESS_GB_TOTAL / MONTHS * P['egress_usd_gb']:,.2f} \\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"\\multicolumn{{3}}{{l}}{{Monthly total (the calculator's ``monthly cost''); $\\times$\\,6 months = "
             f"\\${total_usd:,.2f}}} & \\textbf{{\\${total_usd / MONTHS:,.2f}}} \\\\\n")

# Application e-mail (GRNET asks for: purpose/name/summary, supervisor, laboratory, duration, calculator estimate)
EMAIL = f"""To: helpdesk@aws.grnet.gr
Cc: Prof. George K. Matsopoulos <gmatsopoulos@biomed.ntua.gr>
Subject: Application for AWS resources - US-Foundation (NTUA, Biomedical Engineering Laboratory)
Attachment: proposal.pdf (Detailed Project Document)

Dear GRNET AWS team,

We would like to apply for AWS resources for the following research project.

1. Purpose of the request
   Project name: US-Foundation: Scaling Self-Supervised Ultrasound Foundation Models
   for Generalizable Multi-Task Biometry.
   Summary: We develop a single AI model that automatically places the anatomical
   landmarks behind nine routine ultrasound measurements (fetal biometry, labour
   progression, cardiac and vascular dimensions). It adapts the DINOv2 vision foundation
   model to ultrasound with self-supervised learning on 191,170 unlabeled frames and then
   trains an HRNet-style landmark decoder on 6,768 labeled images; the first version is
   described in our camera-ready paper for the MICCAI 2026 Foundation Model Challenge for
   Ultrasound Biometry. The requested resources fund a six-month programme of
   {sum(j.n for j in J)} GPU jobs that (Q1) re-evaluates the method with leakage-free,
   cine-loop-grouped cross-validation, (Q2) determines how long, at which resolution and
   on how much data the encoder should be adapted, (Q3) scales the encoder from ViT-B
   (87M parameters) to ViT-g (1.14B), and (Q4) optimizes the landmark decoder; code and
   models will be released openly.
   Resources: one Amazon EC2 p4de.24xlarge instance (8x NVIDIA A100-80GB, On-Demand),
   run only in bursts ({node_h:,.0f} instance-hours in total), a t3.large control node,
   a 12-hour p5.4xlarge (H100) pilot, 1 TB Amazon EBS gp3, 1 TB Amazon S3 and
   {EGRESS_GB_TOTAL} GB of data transfer, in US East (N. Virginia), us-east-1, the Region
   with the lowest A100-80GB price.

2. Scientific supervisor
   Prof. George K. Matsopoulos, Professor, School of Electrical and Computer Engineering,
   National Technical University of Athens.

3. Institution / laboratory
   Biomedical Engineering Laboratory, School of Electrical and Computer Engineering,
   National Technical University of Athens (NTUA), Athens, Greece.

4. Duration
   6 months (with the possibility of renewal), from the activation of the credits.

5. Estimated cost (AWS Pricing Calculator, us-east-1, On-Demand)
   ${total_usd:,.2f} for 6 months (${total_usd / MONTHS:,.2f} per month), i.e.
   EUR {total_eur:,.2f} at the ECB reference rate of {P["usd_per_eur_date"]} (1 EUR = {USD_PER_EUR} USD).
   Requested amount: EUR {request_eur:,}.
   AWS Pricing Calculator estimate: <PASTE THE SHARED ESTIMATE LINK HERE>

The attached Detailed Project Document gives the scientific background, the work plan,
the measured timings on which every line of the estimate is based, and the cost-control
measures. Publications resulting from these resources will state that "the resources
were granted with the support of GRNET", and their citation records will be sent to
helpdesk@aws.grnet.gr.

Kind regards,
Alexandros Barmperis
PhD candidate, Biomedical Engineering Laboratory, NTUA
on behalf of the scientific supervisor, Prof. George K. Matsopoulos
amparmperis@biomed.ntua.gr
"""
with open(D("generated", "email_application.txt"), "w") as fh:
    fh.write(EMAIL)

# --------------------------------------------------------------------------- figures (dataviz palette, light)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # reference slots 1-3 (validated all-pairs)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0,
    "ytick.major.size": 0, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def grid_only(ax, axis):
    ax.grid(False)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)


# Fig. scaling: (a) throughput vs encoder size, (b) measured wall-clock per job
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.95), gridspec_kw={"width_ratios": [1.0, 1.15]})
series = [("Phase-1 SSL, 224 px", ("ssl", 224, None), BLUE),
          ("Phase-1 SSL, 518 px", ("ssl", 518, None), ORANGE),
          ("Phase-2 fine-tuning, 518 px", ("phase2", 518, 4), AQUA)]
PARAMS = {bb: bench("ssl", bb, 224)["params_M"] for bb in ("b", "l", "g")}
for label, (kind, res, unf), col in series:
    xs = [PARAMS[bb] for bb in ("b", "l", "g")]
    ys = [bench(kind, bb, res, unf)["img_s"] for bb in ("b", "l", "g")]
    a.plot(xs, ys, color=col, lw=2, solid_capstyle="round", zorder=3, label=label)
    a.scatter(xs, ys, s=44, color=col, edgecolor="white", linewidth=1.8, zorder=4)
    a.annotate(f"{ys[-1]:.0f}" if ys[-1] >= 10 else f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(7, 0),
               textcoords="offset points", va="center", color=INK, fontsize=7)
    a.annotate(f"{ys[0]:.0f}", (xs[0], ys[0]), xytext=(-7, 0), textcoords="offset points", va="center",
               ha="right", color=INK, fontsize=7)
pv = [(PARAMS["l"], M["ssl_vitl_224_epoch_h"]["img_per_s"]), (PARAMS["l"], M["ssl_vitl_518_epoch_h"]["img_per_s"]),
      (PARAMS["l"], 84.4)]
a.scatter(*zip(*pv), s=95, facecolor="none", edgecolor=INK, linewidth=1.0, zorder=5, label="ViT-L in production (logs)")
a.set_xscale("log"), a.set_yscale("log")
a.xaxis.set_major_locator(FixedLocator([PARAMS[bb] for bb in ("b", "l", "g")])), a.xaxis.set_minor_locator(NullLocator())
a.set_xticklabels(["ViT-B/14\n87M", "ViT-L/14\n304M", "ViT-g/14\n1.14B"])
a.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200])), a.yaxis.set_minor_locator(NullLocator())
a.set_yticklabels(["5", "10", "20", "50", "100", "200"])
a.set_xlim(45, 2300), a.set_ylim(3.2, 300)
a.set_ylabel("images / s, 1x A100-80GB (log)")
a.set_title("(a) Throughput vs encoder size", pad=30)
a.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.4, handlelength=1.5,
         columnspacing=0.9, borderaxespad=0.2)
grid_only(a, "y")

jobs = [("Test inference, 5 models x 9 views", test5, False),
        ("SSL epoch, 224 px (191k images)", e224, False),
        ("Frozen-encoder probe", probe, False),
        ("SSL epoch, 518 px", e518, False),
        ("Phase-2 run, 4 blocks unfrozen", p2, False),
        ("Phase-2 run, all 24 blocks", fullft, False),
        ("Paper SSL recipe, 100 + 4 epochs", M["ssl_paper_recipe_h"], False),
        ("Full-res. SSL, 100 ep (projected)", 100 * e518, True)]
yy = range(len(jobs))
for y, (lab, h, proj) in zip(yy, jobs):
    b.barh(y, h, height=0.56, color=BLUE, alpha=0.40 if proj else 1.0, zorder=3)
    b.text(h * 1.1, y, f"{h:,.1f} h" if h < 10 else f"{h:,.0f} h", va="center", color=INK, fontsize=7)
b.set_yticks(list(yy)), b.set_yticklabels([j[0].replace(" x ", " × ") for j in jobs], fontsize=7)
b.set_xscale("log"), b.set_xlim(0.3, 3000)
b.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])), b.xaxis.set_minor_locator(NullLocator())
b.set_xticklabels(["1 h", "10 h", "100 h", "1,000 h"])
b.set_title("(b) Measured wall-clock per job, ViT-L, 1x A100", pad=30)
b.spines["left"].set_visible(False)
grid_only(b, "x")
fig.tight_layout(w_pad=1.0)
os.makedirs(D("figs"), exist_ok=True)
fig.savefig(D("figs", "fig_scaling.pdf"), bbox_inches="tight")
plt.close(fig)

# Fig. schedule: (a) Gantt of work packages, (b) monthly A100-hours by phase
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
names = list(WPS)[::-1]
for i, wp in enumerate(names):
    s0, e0 = SPANS[wp]
    a.barh(i, e0 - s0, left=s0, height=0.5, color=BLUE, zorder=3)
    a.text(e0 + 0.1, i, f"{sum(j.budget for j in by_wp[wp]):,.0f} A100-h", va="center", fontsize=6.8, color=INK)
a.set_yticks(range(len(names))), a.set_yticklabels([f"{w}  {WPS[w].split(' (')[0]}" for w in names], fontsize=6.8)
a.set_xlim(0, 7.35), a.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), a.set_xticklabels([f"M{m}" for m in range(1, 7)])
for m in range(1, 7):
    a.axvline(m, color=GRID, lw=0.6, zorder=1)
a.grid(False), a.spines["left"].set_visible(False)
a.set_title("(a) Work plan", pad=18)

ms = [r[0] for r in month_rows]
ssl_v, sup_v = [r[1] for r in month_rows], [r[2] for r in month_rows]
b.bar(ms, ssl_v, width=0.56, color=BLUE, zorder=3, label="Phase-1 SSL")
b.bar(ms, sup_v, width=0.56, bottom=ssl_v, color=ORANGE, zorder=3, label="Phase-2, probes, inference",
      edgecolor="white", linewidth=1.5)
top = max(r[1] + r[2] for r in month_rows)
for r in month_rows:
    b.text(r[0], r[1] + r[2] + top * 0.03, f"{r[4]:,.0f}", ha="center", fontsize=6.8, color=INK)
b.set_xticks(ms), b.set_xticklabels([f"M{m}" for m in ms])
b.set_ylabel("budgeted A100-hours")
b.set_ylim(0, top * 1.15)
b.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.4, handlelength=1.2,
         columnspacing=0.9, borderaxespad=0.2)
grid_only(b, "y")
b.set_title("(b) Monthly GPU use", pad=18)
fig.tight_layout(w_pad=1.2)
fig.savefig(D("figs", "fig_schedule.pdf"), bbox_inches="tight")
plt.close(fig)

# Fig. regions: $/GPU-hour for A100-80GB (p4de) and A100-40GB (p4d) where offered
p4de, p4d = P["ec2_usd_per_h"]["p4de.24xlarge"], P["ec2_usd_per_h"]["p4d.24xlarge"]
regs = sorted(set(p4de) | set(p4d), key=lambda r: (p4de.get(r, 1e9), p4d.get(r, 1e9)))
fig, ax = plt.subplots(figsize=(3.3, 3.0))
for y, r in enumerate(regs[::-1]):
    if r in p4d:
        ax.scatter(p4d[r] / 8, y, s=44, color=ORANGE, edgecolor="white", linewidth=1.6, zorder=4)
    if r in p4de:
        ax.scatter(p4de[r] / 8, y, s=44, color=BLUE, edgecolor="white", linewidth=1.6, zorder=4)
ax.scatter([], [], s=44, color=BLUE, label="A100-80GB (p4de)")
ax.scatter([], [], s=44, color=ORANGE, label="A100-40GB (p4d)")
ax.set_yticks(range(len(regs))), ax.set_yticklabels(regs[::-1], fontsize=6.6)
for tl in ax.get_yticklabels():
    if tl.get_text() == REGION:
        tl.set_fontweight("bold"), tl.set_color(INK)
yc = len(regs) - 1 - regs.index(REGION)
ax.annotate(f"${NODE_USD_H / 8:.2f} per GPU-h", (NODE_USD_H / 8, yc), xytext=(8, 0), textcoords="offset points",
            fontsize=6.8, color=INK, va="center")
ax.set_xlabel("On-Demand Linux price, USD per GPU-hour")
ax.xaxis.set_major_locator(FixedLocator([2.5, 3.0, 3.5, 4.0, 4.5])), ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlim(2.5, 5.0)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.4, handletextpad=0.2,
          columnspacing=0.8, borderaxespad=0.2)
grid_only(ax, "x")
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(D("figs", "fig_regions.pdf"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- console summary
print(f"measured {meas:,.0f} A100-h -> budget {budg:,.0f} -> packed {packed:,.0f} -> billed {gpu_h_billed:,.0f} "
      f"({node_h:,.0f} node-h, {node_h / 24:.1f} days)")
for wp, jobs in by_wp.items():
    print(f"  {wp}: measured {sum(j.measured for j in jobs):7,.0f}  budget {sum(j.budget for j in jobs):7,.0f}")
for s in services:
    print(f"  {s[0]:14s} {s[1][:60]:60s} {s[2]:10,.1f} {s[3]:10s} x {s[4]:<9} = ${s[5]:10,.2f}")
print(f"TOTAL ${total_usd:,.2f} = EUR {total_eur:,.2f} (USD/EUR {USD_PER_EUR}); request EUR {request_eur:,}")
print("monthly:", [(r[0], round(r[3]), round(r[4]), round(r[5])) for r in month_rows])
print("ratios:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in R.items()})
print(f"longest job: {longest.label} {longest.unit * SLACK:.0f} h")
