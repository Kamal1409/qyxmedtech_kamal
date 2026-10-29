# Review set: correct the pre-labels (about 1-2 hours)

Goal: measure how well the detector finds sperm heads on videos it never saw.
Correct the boxes; do NOT trust them - the pre-labels come from the detector, and
accepting a wrong box would hide exactly the errors we want to count.

1. In CVAT create a task with one label: Sperm (rectangle).
   Upload everything in `images/`.
2. Task -> Actions -> Upload annotations -> format "YOLO 1.1" -> annotations_yolo11.zip.
3. On each frame fix, in this order of priority:
   - MISSED sperm heads: draw a box (tight around the head, tail excluded - match
     the style of the existing boxes).
   - FALSE boxes on debris, round cells, pin-heads or tails: delete.
   - MERGED / SPLIT boxes: two boxes on one head or one box on two heads.
   Only *heads* count. Tails alone, debris and leukocytes (large round cells) do not.
   If you are unsure whether something is a sperm head, leave it out and note the
   frame in `notes.txt`.
4. Export: Task -> Actions -> Export task dataset -> "YOLO 1.1" (no images needed).
   Unzip into `corrected/` here so the labels are in corrected/obj_train_data/*.txt.
5. Score it:   python -m src.eval_detector --pred data/annotation_review/prelabels \
                   --gt data/annotation_review/corrected/obj_train_data

---

## Context for this batch

This is a second, targeted batch: videos 18 and 23 scored badly on the first batch
(F1 0.10 and 0.00 -- see data/annotation_review). The detector mistakes mucus
strands / debris clumps for sperm heads in these two samples, so expect:

- **Many false boxes to delete** (videos 18/23 pre-labels average ~29/48 boxes per
  frame, but the true count is usually much lower -- check each one).
- **Some real heads the detector missed entirely.**
- These are the two videos this batch exists to fix -- take real care here, since
  a sloppy correction here defeats the purpose.

15 frames were requested per video; you may have fewer for video 23 (its live
segment is short). None of these frames were in the first batch (data/annotation_review) --
safe to correct both batches together in the same CVAT project.

Once corrected, unzip into `data/annotation_review2/corrected/` (same layout as
before) and combine with the first batch when rebuilding the training set.
