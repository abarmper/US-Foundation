# GRNET AWS resource request — US-Foundation

Application for AWS credits through GRNET's cloud programme (applications to
`helpdesk@aws.grnet.gr`; 6 months, renewable; €2,500–€250,000). The deliverables are:

| File | What it is |
|---|---|
| `proposal.pdf` | The *Detailed Project Document* to attach (built from `proposal.tex`). |
| `generated/email_application.txt` | The application e-mail with the five fields GRNET asks for. Paste the calculator link into it. |
| `budget.py` | Single source of truth: measured timings → work plan → AWS credits. Writes `generated/*.tex` (numbers and tables) and `figs/*.pdf`. |
| `data/measured_timings.json` | Per-phase wall-clock parsed from our A100 training logs (`tools/extract_timings.py`). |
| `data/bench_results.jsonl` | ViT-B/L/g throughput on one A100-80GB with the production training step (`tools/bench_scaling.py`). |
| `data/aws_prices.json` | Public AWS On-Demand prices (22 Regions) and the ECB USD/EUR rate (`tools/fetch_aws_prices.py`). |

## Current estimate (prices of 2026-09-25, ECB rate of 2026-09-29)

| | |
|---|---|
| Work plan | 396 single-GPU jobs, 2,920 A100-hours at measured rates |
| Budgeted | +25 % per-job slack → ÷0.90 packing → +20 % contingency = **612 p4de.24xlarge instance-hours** (4,896 A100-hours) |
| Region | `us-east-1` (cheapest A100-80GB, $27.44705/h; `us-west-2` same price) |
| Total | **$17,580.47 = €15,482.58** (1 € = 1.1355 US$) → request **€15,483** |

## Rebuild (run from the repository root)

```bash
python docs/GRNET_AWS/tools/extract_timings.py       # re-parse runs/*/phase{1,2}_*.log
python docs/GRNET_AWS/tools/fetch_aws_prices.py      # refresh prices + ECB rate (do this right before sending)
python docs/GRNET_AWS/budget.py                      # tables, macros, figures, e-mail
cd docs/GRNET_AWS && latexmk -pdf proposal.tex
```

`tools/bench_scaling.py ssl|p2` re-measures throughput (needs one free A100-80GB, ~15 min);
append its JSON lines to `data/bench_results.jsonl` only if you want to replace the current numbers.

## Creating the AWS Pricing Calculator link (the call requires it)

The calculator uses the same public price list, so these entries reproduce the estimate to the cent
(Appendix A of the PDF lists the same numbers):

1. Open <https://calculator.aws/> → **Create estimate**. Set the Region to **US East (N. Virginia)** for every service.
2. **Amazon EC2** → Tenancy *Shared*, OS *Linux*, Workload *Constant usage*, 1 instance, **p4de.24xlarge**,
   Pricing strategy **On-Demand**, usage **102 hours/month**. Set the EBS storage of this entry to **0 GB**
   (EBS is added separately below). Expected: **$2,799.60/month**.
3. **Amazon EC2** (second entry) → same settings, **p5.4xlarge**, **2 hours/month**, EBS 0 GB → **$13.76/month**
   (this is the 12-hour H100 pilot spread over 6 months).
4. **Amazon EC2** (third entry) → **t3.large**, **100 hours/month**, EBS 0 GB → **$8.32/month**.
5. **Amazon Elastic Block Store (EBS)** → 1 volume, **General Purpose SSD (gp3)**, **1,000 GB**, 3,000 IOPS,
   125 MBps, 730 hours/month, no snapshots → **$80.00/month**.
6. **Amazon Simple Storage Service (S3)** → S3 Standard **1,000 GB/month**, **100,000** PUT/COPY/POST/LIST
   requests, **1,000,000** GET/SELECT requests → **$23.90/month**.
7. **Data Transfer** → outbound to the internet, **50 GB/month** → **$4.50/month**.
8. The estimate's monthly total must read **$2,930.08** (× 6 months = $17,580.47).
   Name it "US-Foundation – GRNET AWS", **Save and share**, and paste the link into
   `\CalcLink` in `proposal.tex` and into `generated/email_application.txt`.

If the calculator shows a different number, prices have changed. Re-run `fetch_aws_prices.py` and
`budget.py`, then rebuild the PDF.

## Before sending — to confirm

- [ ] Paste the calculator link (header table of the PDF + e-mail).
- [ ] End date of the NVIDIA Academic Hardware Grant allocation (red marker in Section 4).
- [ ] Whether `github.com/abarmper/US-Foundation` is public (red marker in Section 3). Otherwise
      drop the link or say "available on request".
- [ ] Supervisor approval of the text; Prof. Matsopoulos is the scientific supervisor and is responsible
      for the use of the credits under the call's terms.
- [ ] The call asks for the scientific supervisor's "name and capacity" and the laboratory; both are in the
      header table and in the e-mail.
