"""Fetch the public AWS on-demand prices and the ECB USD/EUR rate used by budget.py.

Sources (no AWS credentials needed):
  * EC2 Linux on-demand, per region: the JSON behind aws.amazon.com/ec2/pricing/on-demand
    (b0.p.awsstatic.com), cross-checked against the official Price List API CSV
    (pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.csv).
  * EBS gp3 / snapshots (same official EC2 CSV), S3 Standard (AmazonS3 offer file),
    internet egress (AWSDataTransfer offer file), all for us-east-1.
  * ECB euro foreign-exchange reference rate (eurofxref-daily.xml).

Re-run right before submitting so the estimate reflects the current price list:
    python docs/GRNET_AWS/tools/fetch_aws_prices.py
"""
import csv
import datetime as dt
import io
import json
import re
import urllib.parse
import urllib.request
import gzip

OUT = "docs/GRNET_AWS/data/aws_prices.json"
REGIONS = {  # AWS pricing-page region label -> region code
    "US East (N. Virginia)": "us-east-1", "US East (Ohio)": "us-east-2", "US West (Oregon)": "us-west-2",
    "US West (N. California)": "us-west-1", "Canada (Central)": "ca-central-1",
    "EU (Frankfurt)": "eu-central-1", "EU (Ireland)": "eu-west-1", "EU (London)": "eu-west-2",
    "EU (Paris)": "eu-west-3", "EU (Stockholm)": "eu-north-1", "EU (Milan)": "eu-south-1",
    "EU (Spain)": "eu-south-2", "EU (Zurich)": "eu-central-2", "Israel (Tel Aviv)": "il-central-1",
    "Middle East (UAE)": "me-central-1", "Asia Pacific (Tokyo)": "ap-northeast-1",
    "Asia Pacific (Seoul)": "ap-northeast-2", "Asia Pacific (Mumbai)": "ap-south-1",
    "Asia Pacific (Singapore)": "ap-southeast-1", "Asia Pacific (Sydney)": "ap-southeast-2",
    "Asia Pacific (Jakarta)": "ap-southeast-3", "South America (Sao Paulo)": "sa-east-1",
}
TYPES = ["p4de.24xlarge", "p4d.24xlarge", "p5.4xlarge", "p5.48xlarge", "g6e.xlarge", "g7e.2xlarge", "t3.large"]
WEB = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/{}/Linux/index.json"
OFFER = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{}/current/us-east-1/index.{}"


def get(url):
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "gzip"}), timeout=300).read()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def ec2_by_region():
    table, published = {t: {} for t in TYPES}, None
    for label, code in REGIONS.items():
        d = json.loads(get(WEB.format(urllib.parse.quote(label))))
        published = d["manifest"]["hawkFilePublicationDate"]
        for v in next(iter(d["regions"].values())).values():
            if v["Instance Type"] in table:
                table[v["Instance Type"]][code] = float(v["price"])
    return table, published


def ec2_csv_us_east_1():
    """GPU specs + EBS prices from the official Price List API CSV (streamed, ~300 MB)."""
    text = io.TextIOWrapper(urllib.request.urlopen(OFFER.format("AmazonEC2", "csv"), timeout=600), encoding="utf-8")
    meta = {}
    for _ in range(5):
        k, v = next(csv.reader([text.readline()]))
        meta[k] = v
    spec, ebs = {}, {}
    for row in csv.DictReader(text):
        if row["TermType"] != "OnDemand":
            continue
        it = row["Instance Type"]
        if (it in TYPES and row["Operating System"] == "Linux" and row["Tenancy"] == "Shared"
                and row["Pre Installed S/W"] == "NA" and row["CapacityStatus"] == "Used"
                and row["MarketOption"] == "OnDemand" and row["License Model"] == "No License required"):
            spec[it] = {"usd_per_h": float(row["PricePerUnit"]), "gpus": row["GPU"], "gpu_memory": row["GPU Memory"],
                        "vcpu": row["vCPU"], "memory": row["Memory"], "storage": row["Storage"]}
        elif row["Product Family"] == "Storage" and row["Volume API Name"] == "gp3":
            ebs["gp3_gb_month"] = float(row["PricePerUnit"])
        elif row["Product Family"] == "Storage Snapshot" and row["usageType"] == "EBS:SnapshotUsage":
            ebs["snapshot_gb_month"] = float(row["PricePerUnit"])
    return meta["Publication Date"], spec, ebs


def offer_price(service, pred):
    d = json.loads(get(OFFER.format(service, "json")))
    for sku, p in d["products"].items():
        if pred(p.get("productFamily"), p["attributes"]):
            for term in d["terms"]["OnDemand"].get(sku, {}).values():
                for dim in term["priceDimensions"].values():
                    if dim.get("beginRange", "0") == "0":
                        return float(dim["pricePerUnit"]["USD"]), d["publicationDate"]


def ecb_usd_per_eur():
    xml = get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml").decode()
    return float(re.search(r"currency='USD' rate='([\d.]+)'", xml).group(1)), re.search(r"time='([\d-]+)'", xml).group(1)


if __name__ == "__main__":
    by_region, web_pub = ec2_by_region()
    csv_pub, spec, ebs = ec2_csv_us_east_1()
    s3, s3_pub = offer_price("AmazonS3", lambda f, a: f == "Storage" and a.get("volumeType") == "Standard"
                             and a.get("storageClass") == "General Purpose")
    egress, dt_pub = offer_price("AWSDataTransfer", lambda f, a: a.get("fromLocation") == "US East (N. Virginia)"
                                 and a.get("toLocation") == "External" and a.get("transferType") == "AWS Outbound")
    fx, fx_date = ecb_usd_per_eur()
    out = {
        "retrieved": dt.date.today().isoformat(),
        "ec2_published": web_pub, "ec2_csv_published": csv_pub, "s3_published": s3_pub, "egress_published": dt_pub,
        "ec2_usd_per_h": by_region, "ec2_spec_us_east_1": spec,
        "ebs_gp3_usd_gb_month": ebs["gp3_gb_month"], "ebs_snapshot_usd_gb_month": ebs["snapshot_gb_month"],
        "s3_standard_usd_gb_month": s3, "egress_usd_gb": egress,
        "usd_per_eur": fx, "usd_per_eur_date": fx_date, "usd_per_eur_source": "ECB euro foreign exchange reference rate",
    }
    for it, v in spec.items():  # the two sources must agree
        assert abs(v["usd_per_h"] - by_region[it]["us-east-1"]) < 1e-6, it
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "ec2_usd_per_h"}, indent=1))
