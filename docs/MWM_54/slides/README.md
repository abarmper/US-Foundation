# FUB 2026 presentation

13 slides (no backups), 16:9, built from the paper (`../main.tex`), paced for
a **10-minute talk**: the front half is picture-led, key terms are bolded, and
the hyperparameter detail lives in the speaker notes and in `talk_script.md`'s
Q&A appendix.

## Files

| File | |
|---|---|
| `31_annapan_54_presentation.pptx` | The deck. Filename follows the organizers' rule (`rank + team + paper ID`) — **confirm the rank and account before submitting**. |
| `build_slides.py` | Builds the deck. All content lives in `SLIDES`; it also emits `outline.md`, so deck and mirror cannot drift. Bullet/caption text supports `**bold**`, `*italic*`, and `__accent__` inline markup. |
| `make_vit_schematic.py` | Draws `img/vit_schematic.png`, the patch-tokenization diagram the paper does not have. |
| `make_extra_schematics.py` | Draws the three other slide-only figures: `task_imbalance.png` (slide 2), `metric_equal_weight.png` (slide 3), `method_overview.png` (slide 4). Per-task labeled counts come from `data/splits/train_val_split_keys.json`. |
| `talk_script.md` | Word-for-word 10-minute script with per-slide timestamps, plus a Q&A appendix holding what the deleted backup slides used to carry. |
| `talk_script.docx` | The same script as a Word document, rendered by `talk_script_to_docx.py` — edit the .md and re-run that script, not the .docx. |
| `outline.md` | Text mirror: every slide's bullets plus speaker notes. |
| `img/` | Rasterized paper figures, the two `fig2` crops, and the schematics. |

Speaker notes are in the .pptx notes slides as well as in `outline.md`; where a
bullet was trimmed for time, the removed numbers were moved into the notes.

## Rebuilding

```bash
pip install python-pptx                    # plus lxml, Pillow, XlsxWriter

# 1. rasterize the paper's figures at 200 dpi
cd ../figures
for f in *.pdf; do
    pdftoppm -r 200 -png -singlefile "$f" "../slides/img/${f%.pdf}"
done
cd ../slides

# 2. split fig2 into the neck half and the heads half.
#    The cut at x=3050 falls in the gutter between the FUSION panel and the
#    task-routing box — verify visually if the figure is ever regenerated.
convert img/fig2_hrnet_neck.png -crop 3060x1506+0+0    +repage img/fig2_left_neck.png
convert img/fig2_hrnet_neck.png -crop  742x1506+3050+0 +repage img/fig2_right_heads.png

# 3. draw the schematics, then build
python3 make_vit_schematic.py
python3 make_extra_schematics.py
python3 build_slides.py
```

## Notes on content

- The deck presents the **paper's** design: the multilevel neck with 148×148
  heatmaps (`fig2` unmodified) and the paper's Phase-2 hyperparameters. The
  model actually shipped in the final-test Docker image was the single-level
  variant with 128×128 heatmaps under the legacy "simple" recipe — that one is
  documented in `../materials/structured_documentation_filled.md`. The two
  documents travel in the same materials package, so the difference is worth
  knowing about if anyone asks.
- Slide 10's results are the **official validation-phase** evaluation (619
  images), which is what the paper reports. If the organizers expect
  hidden-test figures, take them from the preliminary results table on
  CodaBench competition 17560.
- Every number was cross-checked against `../main.tex` and
  `../../latex_template/suppl_mat/scores.txt`. Two exceptions: the Phase-1
  EMA momentum in slide 7's notes (0.994 → 1.0) comes from
  `gubiometry/config.py:128-129`, and the per-task labeled counts on the
  slide-2/3 charts come from `data/splits/train_val_split_keys.json`
  (they sum to 6,743 — the paper's 6,768 total includes a handful of rows
  excluded from the splits — but match every count the paper states:
  AoP 4,000, HC 999, PSAX 49, IVC 38).
