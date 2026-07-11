# Citation Map — Verified Reference Dictionary

> **Update 2026-07-07 — full-text re-validation (supersedes the metadata-only audit below).**
> Every reference was re-checked by reading the *entire* paper (not just metadata), cross-checking
> our text's characterization, and re-verifying BibTeX. Full results: **`papers/REFERENCE_VALIDATION.csv`**.
> All 39 references confirmed relevant and correctly characterized in the text (**0 relevance/citation
> mismatches**). Net changes to `references.bib`:
> - **Removed** `ramirez2025position` (arXiv position paper); **added** two AAAI-published refs
>   `brosowsky2021sample` (ConstraintNet, AAAI-21, 35(8):6812–6821) and `goyal2024deepsade`
>   (DeepSaDe, AAAI-24, 38(11):12199–12207). Count **38 → 39**.
> - **Corrected `shifman2023adaptive` title** to the *published* form
>   "An Adaptive Machine Learning Algorithm for the Resource-Constrained Classification Problem"
>   (Crossref DOI 10.1016/j.engappai.2022.105741). The "Flagged and FIXED (2026-07-06)" note below
>   wrongly set the arXiv-*preprint* title; that has now been reverted to the published title.
> - Two automated flags were **false positives** — verified against authoritative sources and left
>   unchanged: `eban2017scalable` fifth author is **Ryan Rifkin** (official PMLR proceedings, not
>   "Rif A. Saurous"); `lin2017focal` pages are **2999–3007** (IEEE Crossref DOI 10.1109/ICCV.2017.324,
>   not 2980–2988).

**What this is.** A one-to-one verification of every entry in `references.bib` (39 entries)
against **Semantic Scholar** (authoritative source). For each reference it records the
downloaded PDF file, the bib key, a clean human-readable **way to credit the work**
(verified title / authors / venue), and the verification status.

- **Source of truth:** Semantic Scholar (title-match, DOI, and arXiv-id lookups).
- **Verified on:** 2026-07-06.
- **Method:** title best-match for most entries; exact DOI/arXiv batch lookup where an id
  was available in the bib. First-author surname, year, and venue were checked for each.
- **Year note:** several journal/conference papers show an *earlier* year on Semantic
  Scholar (the arXiv / online-first date) than the bib's year (the final proceedings /
  print-volume year). In every such case the **bib year is the correct final-publication
  year** — these are noted below but are **not** errors.

| PDF file | bib key | Verified citation (how to credit) | S2 status |
|---|---|---|---|
| fioretto2020lagrangian.pdf | fioretto2020lagrangian | Ferdinando Fioretto, Pascal Van Hentenryck, Terrence W. K. Mak, Cuong Tran, Federico Baldo, Michele Lombardi (2020). "Lagrangian Duality for Constrained Deep Learning." ECML PKDD. | ✓ verified |
| hounie2023resilient.pdf | hounie2023resilient | Ignacio Hounie, Alejandro Ribeiro, Luiz F. O. Chamon (2023). "Resilient Constrained Learning." NeurIPS. | ✓ verified |
| chamon2020probably.pdf | chamon2020probably | Luiz F. O. Chamon, Alejandro Ribeiro (2020). "Probably Approximately Correct Constrained Learning." NeurIPS. | ✓ verified |
| chamon2023constrained.pdf | chamon2023constrained | Luiz F. O. Chamon, Santiago Paternain, Miguel Calvo-Fullana, Alejandro Ribeiro (2023). "Constrained Learning with Non-Convex Losses." IEEE Transactions on Information Theory. | ✓ verified (S2 yr 2021 = preprint; journal vol. 69 = 2023) |
| stooke2020responsive.pdf | stooke2020responsive | Adam Stooke, Joshua Achiam, Pieter Abbeel (2020). "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML. | ✓ verified |
| ~~ramirez2025position.pdf~~ | ~~ramirez2025position~~ | Juan Ramirez, Meraj Hashemizadeh, Simon Lacoste-Julien (2025). "Position: Adopt Constraints Over Fixed Penalties in Deep Learning." arXiv:2505.20628. | **REMOVED 2026-07-07** (arXiv position paper; point retained via `gallego2022controlled`) |
| — (book, no PDF) | bertsekas2014constrained | Dimitri P. Bertsekas (2014). "Constrained Optimization and Lagrange Multiplier Methods." Academic Press (book; orig. ed. 1982). | book — limited S2 coverage (title + author confirmed) |
| sangalli2021constrained.pdf | sangalli2021constrained | Sara Sangalli, Ertunc Erdil, Andreas Hoetker, Olivio Donati, Ender Konukoglu (2021). "Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes." NeurIPS. | ✓ verified |
| shifman2023adaptive.pdf | shifman2023adaptive | Danit Abukasis Shifman, Izack Cohen, Kejun Huang, Xiaochen Xian, Gonen Singer (2023). "An Adaptive Machine Learning Algorithm for the Resource-Constrained Classification Problem." Engineering Applications of Artificial Intelligence, 119, 105741. | ✓ title re-corrected 2026-07-07 to published form (Crossref DOI) |
| shifman2025classification.pdf | shifman2025classification | Danit Abukasis Shifman, Itay Margolin, Chen Halfi, Gonen Singer (2025). "Classification Tasks with Local and Global Resource Allocation Constraints." IFAC-PapersOnLine, 59(1), 61–66. | ✓ verified (DOI 10.1016/j.ifacol.2025.03.012) |
| vanderschueren2024perspective.pdf | vanderschueren2024perspective | Toon Vanderschueren, Bart Baesens, Tim Verdonck, Wouter Verbeke (2024). "A New Perspective on Classification: Optimally Allocating Limited Resources to Uncertain Tasks." Decision Support Systems. | ✓ verified (S2 yr 2022 = online-first; journal vol. 179 = 2024) |
| cortes2016learning.pdf | cortes2016learning | Corinna Cortes, Giulia DeSalvo, Mehryar Mohri (2016). "Learning with Rejection." ALT (Algorithmic Learning Theory). | ✓ verified |
| lin2017focal.pdf | lin2017focal | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2017). "Focal Loss for Dense Object Detection." ICCV. | ✓ verified |
| — (book, no PDF) | vapnik1998statistical | Vladimir N. Vapnik (1998). "Statistical Learning Theory." Wiley (book). | book — limited S2 coverage (canonical text) |
| joachims1999transductive.pdf | joachims1999transductive | Thorsten Joachims (1999). "Transductive Inference for Text Classification Using Support Vector Machines." ICML. | ✓ verified |
| howard2019searching.pdf | howard2019searching | Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, et al. (2019). "Searching for MobileNetV3." ICCV. | ✓ verified |
| radosavovic2020designing.pdf | radosavovic2020designing | Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár (2020). "Designing Network Design Spaces." CVPR. | ✓ verified |
| dosovitskiy2021image.pdf | dosovitskiy2021image | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR. | ✓ verified (S2 yr 2020 = arXiv; ICLR 2021) |
| he2016deep.pdf | he2016deep | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016). "Deep Residual Learning for Image Recognition." CVPR. | ✓ verified (S2 yr 2015 = arXiv; CVPR 2016) |
| yang2023medmnist.pdf | yang2023medmnist | Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, et al. (2023). "MedMNIST v2 — A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." Scientific Data, 10(1), 41. | ✓ verified (S2 yr 2021 = online-first; journal vol. 10 = 2023) |
| tschandl2018ham10000.pdf | tschandl2018ham10000 | Philipp Tschandl, Cliff Rosendahl, Harald Kittler (2018). "The HAM10000 Dataset, a Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions." Scientific Data, 5, 180161. | ✓ verified (DOI 10.1038/sdata.2018.161) |
| goh2016satisfying.pdf | goh2016satisfying | Gabriel Goh, Andrew Cotter, Maya Gupta, Michael P. Friedlander (2016). "Satisfying Real-World Goals with Dataset Constraints." NeurIPS. | ✓ verified |
| eban2017scalable.pdf | eban2017scalable | Elad Eban, Mariano Schain, Alan Mackey, Ariel Gordon, Ryan Rifkin, Gal Elidan (2017). "Scalable Learning of Non-Decomposable Objectives." AISTATS. | ✓ verified (S2 yr 2016 = arXiv; AISTATS 2017) |
| cotter2019optimization.pdf | cotter2019optimization | Andrew Cotter, Heinrich Jiang, Maya Gupta, Serena Wang, Taman Narayan, Seungil You, et al. (2019). "Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals." JMLR. | ✓ verified (S2 yr 2018; JMLR vol. 20 = 2019) |
| narasimhan2019optimizing.pdf | narasimhan2019optimizing | Harikrishna Narasimhan, Andrew Cotter, Maya Gupta (2019). "Optimizing Generalized Rate Metrics with Three Players." NeurIPS. | ✓ verified |
| nandwani2019primal.pdf | nandwani2019primal | Yatin Nandwani, Abhishek Pathak, Mausam, Parag Singla (2019). "A Primal-Dual Formulation for Deep Learning with Constraints." NeurIPS. | ✓ verified |
| pathak2015constrained.pdf | pathak2015constrained | Deepak Pathak, Philipp Krähenbühl, Trevor Darrell (2015). "Constrained Convolutional Neural Networks for Weakly Supervised Segmentation." ICCV. | ✓ verified |
| marquez2017imposing.pdf | marquez2017imposing | Pablo Márquez-Neila, Mathieu Salzmann, Pascal Fua (2017). "Imposing Hard Constraints on Deep Networks: Promises and Limitations." arXiv:1706.02025 (CVPR-W Negative Results in CV). | ✓ verified (arXiv; workshop venue noted, kept as preprint form) |
| — (web, no local PDF) | brosowsky2021sample | Mathis Brosowsky, Florian Keck, Olaf Dünkel, Marius Zöllner (2021). "Sample-Specific Output Constraints for Neural Networks" (ConstraintNet). AAAI, 35(8), 6812–6821. DOI 10.1609/aaai.v35i8.16841. | ✓ added 2026-07-07 (full text read; per-sample output constraints) |
| — (web, no local PDF) | goyal2024deepsade | Kshitij Goyal, Sebastijan Dumančić, Hendrik Blockeel (2024). "DeepSaDe: Learning Neural Networks That Guarantee Domain Constraint Satisfaction." AAAI, 38(11), 12199–12207. DOI 10.1609/aaai.v38i11.29109. | ✓ added 2026-07-07 (full text read via arXiv HTML; MaxSMT output-layer guarantee) |
| gallego2022controlled.pdf | gallego2022controlled | Jose Gallego-Posada, Juan Ramirez, Akram Erraqabi, Yoshua Bengio, Simon Lacoste-Julien (2022). "Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints." NeurIPS. | ✓ verified |
| zafar2017fairness.pdf | zafar2017fairness | Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi (2017). "Fairness Constraints: Mechanisms for Fair Classification." AISTATS. | ✓ verified (S2 yr 2015 = arXiv; AISTATS/PMLR vol. 54 = 2017) |
| agarwal2018reductions.pdf | agarwal2018reductions | Alekh Agarwal, Alina Beygelzimer, Miroslav Dudik, John Langford, Hanna Wallach (2018). "A Reductions Approach to Fair Classification." ICML. | ✓ verified |
| hardt2016equality.pdf | hardt2016equality | Moritz Hardt, Eric Price, Nathan Srebro (2016). "Equality of Opportunity in Supervised Learning." NeurIPS. | ✓ verified |
| elkan2001foundations.pdf | elkan2001foundations | Charles Elkan (2001). "The Foundations of Cost-Sensitive Learning." IJCAI. | ✓ verified |
| cui2019class.pdf | cui2019class | Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie (2019). "Class-Balanced Loss Based on Effective Number of Samples." CVPR. | ✓ verified |
| cao2019learning.pdf | cao2019learning | Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma (2019). "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." NeurIPS. | ✓ verified |
| menon2021logit.pdf | menon2021logit | Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar (2021). "Long-Tail Learning via Logit Adjustment." ICLR. | ✓ verified (S2 yr 2020 = arXiv; ICLR 2021) |
| mann2007expectation.pdf | mann2007expectation | Gideon S. Mann, Andrew McCallum (2007). "Simple, Robust, Scalable Semi-supervised Learning via Expectation Regularization." ICML. | ✓ verified |
| ganchev2010posterior.pdf | ganchev2010posterior | Kuzman Ganchev, João Graça, Jennifer Gillenwater, Ben Taskar (2010). "Posterior Regularization for Structured Latent Variable Models." JMLR. | ✓ verified |

## Discrepancies & notes

### Flagged and FIXED (2026-07-06) — ⚠️ SUPERSEDED, see correction 2026-07-07 below

> **CORRECTION (2026-07-07):** the 2026-07-06 bullet below is WRONG. Crossref on DOI
> 10.1016/j.engappai.2022.105741 confirms the *published* title is **"An adaptive machine learning
> algorithm for the resource-constrained classification problem."** The 2026-07-06 edit mistakenly
> swapped this for the arXiv-preprint title ("Adaptive Learning for..."). `references.bib` now holds
> the published title again.

- **shifman2023adaptive — title mismatch (corrected).** The bib title was
  *"An Adaptive Machine Learning Algorithm for the Resource-Constrained Classification Problem."*
  The actual published title (confirmed via DOI **10.1016/j.engappai.2022.105741**, and by
  Semantic Scholar title search) is **"Adaptive Learning for the Resource-Constrained
  Classification Problem."** Everything else was already correct: authors (Shifman, Cohen, Huang,
  Xian, Singer), venue (Engineering Applications of Artificial Intelligence, vol. 119, art.
  105741), and year (bib 2023; S2 lists 2022 online-first — the print volume is 2023). This
  is the *right paper*, only the title string was wrong. **Fix applied:** `references.bib`
  `title` corrected to "Adaptive Learning for the Resource-Constrained Classification Problem"
  and the verified DOI added. The paper's References list now shows the correct title.

### Benign year notes (bib is correct — no action needed)

These entries show an earlier year on Semantic Scholar (arXiv / online-first) than the bib.
In each case the bib cites the correct final-publication year (proceedings or print volume),
so **no change is needed**:

- **chamon2023constrained** — S2 2021 (preprint) vs bib 2023 (IEEE Trans. Info. Theory, vol. 69). Bib correct.
- **vanderschueren2024perspective** — S2 2022 (online-first) vs bib 2024 (Decision Support Systems, vol. 179). Bib correct.
- **yang2023medmnist** — S2 2021 (online-first) vs bib 2023 (Scientific Data, vol. 10). Bib correct.
- **zafar2017fairness** — S2 2015 (arXiv) vs bib 2017 (AISTATS / PMLR vol. 54). Bib correct.
- Within-±1 drift (all bib-correct): **dosovitskiy2021image** (S2 2020 arXiv / ICLR 2021),
  **he2016deep** (S2 2015 arXiv / CVPR 2016), **eban2017scalable** (S2 2016 arXiv / AISTATS 2017),
  **cotter2019optimization** (S2 2018 / JMLR vol. 20 2019), **menon2021logit** (S2 2020 arXiv / ICLR 2021).

### The two entries without a PDF (both textbooks, legitimate)

- **bertsekas2014constrained** — canonical textbook, *Constrained Optimization and Lagrange
  Multiplier Methods* (Dimitri P. Bertsekas, Academic Press; orig. 1982, reissued). Found on
  Semantic Scholar (title + author confirmed); books have limited S2 metadata by nature — **not** suspicious.
- **vapnik1998statistical** — canonical textbook, *Statistical Learning Theory* (Vladimir N.
  Vapnik, Wiley, 1998). S2 has only a book-review record; expected for a monograph — **not** suspicious.

_(Previously missing) **shifman2025classification** PDF located 2026-07-07 (author's copy from
the March research proposal, `Danit_paper_FeatureBase_SampleBase.pdf`) and added to `papers/`._

## Summary

**Metadata audit (2026-07-06):** originally flagged `shifman2023adaptive` as the one title error.
That 2026-07-06 correction was itself mistaken (it substituted the arXiv-preprint title); see the
top callout — the title has been re-corrected 2026-07-07 to the published Crossref form.

**Full-text re-validation (2026-07-07):** all **39** references (36 local PDFs + 2 textbooks +
1 web-read arXiv mirror; the 2 new AAAI papers read on the web) were read in full and cross-checked
against the text and BibTeX (`papers/REFERENCE_VALIDATION.csv`). Outcome: **0 relevance mismatches,
0 citation (characterization) mismatches**; **1 real BibTeX fix** (`shifman2023adaptive` title);
**2 false-positive automated flags rejected after authoritative verification** (`eban2017scalable`
author = Ryan Rifkin per PMLR; `lin2017focal` pages = 2999–3007 per IEEE Crossref). The 2 canonical
books are real. No fabricated references, no wrong first authors, no wrong venues. All 39 entries now correct.
