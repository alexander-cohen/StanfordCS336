Alexander Cohen's work through of CS336

# CS336 — Language Modeling from Scratch

**Spring 2026 · Stanford · collected August 31, 2026**

Open [START_HERE.html](START_HERE.html) for the clickable course index, or use this README. Source: [official course site](https://cs336.stanford.edu/).

## Start studying

1. Follow the schedule below. Python lectures have generated offline reading views, original source, cached execution traces, and images; PDF lectures are unchanged.
2. Read each assignment handout and its repository README before setup. All five repositories include starter code, tests, fixtures, dependency lockfiles, `AGENTS.md`, and `CLAUDE.md`. No solutions were generated and no training jobs were run.
3. Use the [reading index](READINGS.md), organized by lecture and assignment. The bundle contains **191 distinct paper/book PDFs** and **110 web text snapshots**. These are references cited by the materials, not a claim that every paper is required reading.
4. Watch the [2026 course recordings](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) online. Video files are not included.

## Course schedule

Dates and deadlines are from the published 2026 schedule; use your own pace for self-study. Readings for each lecture are linked in the last column.

| Lecture | Date | Topic | Local material | Assignment milestones | References |
|---|---|---|---|---|---|
| 1 | Mon March 30 | Overview, tokenization [Percy] | [Offline notes](lectures/offline-notes/lecture_01.html) · [Python source](lectures/lecture_01.py) | Assignment 1 out | [Readings](READINGS.md#lecture-1) |
| 2 | Wed April 1 | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) [Percy] | [Offline notes](lectures/offline-notes/lecture_02.html) · [Python source](lectures/lecture_02.py) · [Recording notes](lectures/offline-notes/lecture_02_recording.html) | — | [Readings](READINGS.md#lecture-2) |
| 3 | Mon April 6 | Architectures, hyperparameters [Tatsu] | [Slides (PDF)](lectures/lecture_03.pdf) | — | [Readings](READINGS.md#lecture-3) |
| 4 | Wed April 8 | Attention alternatives and mixture of experts [Tatsu] | [Slides (PDF)](lectures/lecture_04.pdf) | — | [Readings](READINGS.md#lecture-4) |
| 5 | Mon April 13 | GPUs, TPUs [Tatsu] | [Slides (PDF)](lectures/lecture_05.pdf) | — | [Readings](READINGS.md#lecture-5) |
| 6 | Wed April 15 | Kernels, Triton [Percy] | [Offline notes](lectures/offline-notes/lecture_06.html) · [Python source](lectures/lecture_06.py) | Assignment 1 due Assignment 2 out | [Readings](READINGS.md#lecture-6) |
| 7 | Mon April 20 | Parallelism [Percy] | [Offline notes](lectures/offline-notes/lecture_07.html) · [Python source](lectures/lecture_07.py) | — | [Readings](READINGS.md#lecture-7) |
| 8 | Wed April 22 | Parallelism [Tatsu] | [Slides (PDF)](lectures/lecture_08.pdf) | — | [Readings](READINGS.md#lecture-8) |
| 9 | Mon April 27 | Scaling laws [Tatsu] | [Slides (PDF)](lectures/lecture_09.pdf) | — | [Readings](READINGS.md#lecture-9) |
| 10 | Wed April 29 | Inference [Percy] | [Offline notes](lectures/offline-notes/lecture_10.html) · [Python source](lectures/lecture_10.py) | Assignment 2 due Assignment 3 out | [Readings](READINGS.md#lecture-10) |
| 11 | Mon May 4 | Scaling laws [Tatsu] | [Slides (PDF)](lectures/lecture_11.pdf) | — | [Readings](READINGS.md#lecture-11) |
| 12 | Wed May 6 | Evaluation [Percy] | [Offline notes](lectures/offline-notes/lecture_12.html) · [Python source](lectures/lecture_12.py) | Assignment 3 due Assignment 4 out | [Readings](READINGS.md#lecture-12) |
| 13 | Mon May 11 | Data (sources, datasets) [Percy] | [Offline notes](lectures/offline-notes/lecture_13.html) · [Python source](lectures/lecture_13.py) | — | [Readings](READINGS.md#lecture-13) |
| 14 | Wed May 13 | Data (filtering, deduplication, mixing, synthetic data) [Percy] | [Offline notes](lectures/offline-notes/lecture_14.html) · [Python source](lectures/lecture_14.py) | — | [Readings](READINGS.md#lecture-14) |
| 15 | Mon May 18 | Mid/post-training (SFT/RLHF) [Tatsu] | [Slides (PDF)](lectures/lecture_15.pdf) | — | [Readings](READINGS.md#lecture-15) |
| 16 | Wed May 20 | Post-training - RLVR [Tatsu] | [Slides (PDF)](lectures/lecture_16.pdf) | Assignment 4 due Assignment 5 out | [Readings](READINGS.md#lecture-16) |
| — | Mon May 25 | No class (Memorial Day) | — | — | — |
| 17 | Wed May 27 | Alignment - multimodality [Percy] | [Offline notes](lectures/offline-notes/lecture_17.html) · [Python source](lectures/lecture_17.py) | — | [Readings](READINGS.md#lecture-17) |
| 18 | Mon June 1 | Guest lecture: Daniel Selsam | No posted material | — | — |
| 19 | Wed June 3 | Guest lecture: Dan Fu | No posted material | Assignment 5 due | — |

## Assignments

| Assignment | Release → due (2026) | Handout | Setup and code |
|---|---|---|---|
| 1. Basics | Mar 30 → Apr 15 | [PDF](assignments/assignment1-basics/cs336_assignment1_basics.pdf) | [README](assignments/assignment1-basics/README.md) |
| 2. Systems | Apr 15 → Apr 29 | [PDF](assignments/assignment2-systems/cs336_assignment2_systems.pdf) | [README](assignments/assignment2-systems/README.md) |
| 3. Scaling | Apr 29 → May 6 | [PDF](assignments/assignment3-scaling/cs336_assignment3_scaling.pdf) | [README](assignments/assignment3-scaling/README.md) |
| 4. Data | May 6 → May 20 | [PDF](assignments/assignment4-data/cs336_assignment4_data.pdf) | [README](assignments/assignment4-data/README.md) |
| 5. Alignment and reasoning RL | May 20 → Jun 3 | [PDF](assignments/assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf) | [README](assignments/assignment5-alignment/README.md) |

The [optional safety/RLHF supplement](assignments/assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) uses the current repository's 2026 file; the course site's 2025 filename is stale.

## What is where

- `lectures/`: original slides, Python lectures, images, cached traces, recording version, and `offline-notes/`.
- `assignments/`: five intact course repositories, including their bundled sample data and test weights.
- `readings/papers/` and `readings/web/`: downloaded papers and simplified article/tutorial snapshots. Original paper filenames or arXiv IDs are preserved.
- `code/`: the lecture library source (`edtrace` 0.1.15, pinned by the course lockfile) and individually linked code files. General third-party libraries are links, not full mirrors.
- `resources/`: [AI policy](resources/AI-policy.pdf), [compute guide](resources/compute-guide.pdf), source snapshot, provenance, file checksums, and manifests.

## Offline use and limitations

The static lecture notes and PDFs open directly after unzipping. The static notes preserve the recorded text and figures; use the original Python and trace viewer for code execution details. To view interactive traces, run `python3 serve_course.py` **from this bundle's folder**, then open [Lecture 1's local viewer](http://127.0.0.1:8000/lectures/?trace=lecture_01). Stop the server with Ctrl-C. The helper serves this folder on your computer only and uses local viewer dependencies; it does not execute assignment or lecture code.

Reading snapshots contain text, not complete website mirrors; external images and interactive features may be absent. Current “stable” documentation can be newer than the course: use repository lockfiles for execution. Assignment dependencies, full training corpora, large model weights, GPU compute, and hosted services need separate setup. Assignment 3's student API requires authorized Stanford access; its README includes a non-student path. Guest lectures 18–19 have no materials linked in the course schedule.

See [ACCESS_NOTES.md](ACCESS_NOTES.md) for download failures, stale links, and resources left online. Course source files and licenses are preserved; some upstream README labels still say 2025. This is a personal-study collection, not a new license to redistribute the originals. Follow each source's terms and the included course AI policy.
