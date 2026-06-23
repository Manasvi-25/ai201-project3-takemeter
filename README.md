# TakeMeter
 
A fine-tuned text classifier that identifies discourse types in Indian music communities on Reddit. Given a post or comment from r/MusicIndia, r/india, or r/LetsTalkMusic, the model predicts whether it is a `hot_take`, `reaction`, `recommendation`, or `argument`.
 
---
 
## Community
 
**r/MusicIndia**, **r/india** (music threads), and **r/LetsTalkMusic**
 
These communities were chosen because music discourse there is genuinely varied in type and intent. The same artist or song can generate a direct recommendation, an emotional reaction, a bold controversial opinion, or a pushback against someone else's take — all in the same thread. That range makes classification meaningful. Indian music communities also have a distinctive texture: bilingual posts, Bollywood vs. indie debates, regional artist discussions, and strong opinions about production quality and authenticity, all of which produce a wide variety of post types that map cleanly onto the four labels.
 
---
 
## Label Taxonomy
 
**`hot_take`** — a bold or controversial opinion about an artist, song, or genre stated confidently without supporting evidence or reasoning.
 
> Example 1: "I know she has a strong lineage and following, but I can't enjoy her singing. There seems to be a lack of control and a certain kind of fake depth in her voice."
>
> Example 2: "Mainstream music is totally shit nowadays. It feels like beats are made first and lyrics are forced into it."
 
---
 
**`reaction`** — an immediate emotional or expressive response to a song, artist, or release, with little to no argument or recommendation.
 
> Example 1: "Moon Child, my gosh, I could listen to that song all day, every day."
>
> Example 2: "Woah tyty tamil and Hindi lyrics brings titli to mind haha"
 
---
 
**`recommendation`** — a post primarily aimed at suggesting music to others, whether a single artist, a list, or a curated playlist, with or without brief reasoning.
 
> Example 1: "Tamanna by Yawar Abdal. 3 languages all beautifully woven together. Rait Zara Si from Atrangi Re — the Tamil part hits differently."
>
> Example 2: "If you're into folksy, When Chai Met Toast. If you're into rock-ish stuff, Shine A Light On Me by Raghav Meattle."
 
---
 
**`argument`** — a post that directly responds to or challenges another person's opinion, agreeing or disagreeing with reasoning.
 
> Example 1: "Totally agree man, mainstream music is totally shit nowadays. It feels like beats are first made and lyrics are made to fit forcibly into it. It sucks."
>
> Example 2: "I disagree that AR Rahman has declined. Rockstar and Highway both have incredible scores that people sleep on."
 
---
 
## Dataset
 
### Source and Collection
 
Posts and comments were collected manually from public threads on r/MusicIndia, r/india, and r/LetsTalkMusic. Threads were selected to cover a range of discourse types: recommendation threads ("suggest me Indian indie"), debate threads (AI music ban announcement, Olivia Rodrigo album discussion), reaction threads ("all time favorite song"), and opinion threads ("do people develop a curated music taste"). Comments were copied into a CSV with `text`, `label`, and `notes` columns. 50 additional synthetic examples were generated using Claude to address class imbalance, reviewed for quality and consistency with label definitions before inclusion.
 
### Labeling Process
 
Each example was labeled using the definitions in planning.md. Posts were read individually and assigned one label based on their primary intent. When a post sat between two labels, the decision rules from the Hard Edge Cases section were applied. A `notes` column tracked any cases that required a judgment call.
 
### Label Distribution
 
| Label | Count | Percentage |
|---|---|---|
| hot_take | 87 | 34.8% |
| recommendation | 57 | 22.8% |
| reaction | 56 | 22.4% |
| argument | 50 | 20.0% |
| **Total** | **250** | **100%** |
 
No single label exceeds 70% of the dataset.
 
### Difficult Annotation Cases
 
**Case 1:** "Anuv Jain not so good"
Could be `hot_take` (bold opinion) or `argument` (direct pushback in a thread). Labeled `argument` — it appears as a direct reply to someone recommending Anuv Jain, so the primary intent is pushback against a specific claim rather than an unprompted opinion.
 
**Case 2:** "Listen to it when it rains"
Could be `recommendation` (suggesting how to listen) or `hot_take` (asserting an opinion about the song). Labeled `hot_take` — it is not directing someone to a song, it is making a claim about the best listening context. No music is being recommended.
 
**Case 3:** "Love Demons remind me of old 70s Bollywood soundtracks for some reason."
Could be `reaction` (personal impression) or `hot_take` (comparative claim about the song's sound). Labeled `hot_take` — it makes a specific comparative claim about the song's sound rather than just expressing a feeling about it.
 
---
 
## Fine-Tuning Pipeline
 
**Base model:** `distilbert-base-uncased` (HuggingFace)
**Training platform:** Google Colab (T4 GPU)
 
### Training Setup
 
The model was fine-tuned using HuggingFace `Trainer` with the following final configuration:
 
- `num_train_epochs`: 7
- `per_device_train_batch_size`: 8
- `learning_rate`: 3e-5
- `weight_decay`: 0.01
- `warmup_steps`: 50
### Key Training Decisions
 
**Batch size reduced from 16 to 8:** The default batch size of 16 caused the model to collapse onto the majority class (`hot_take`) after only a few epochs. Reducing to 8 increased gradient update frequency and helped the model learn minority class boundaries more gradually.
 
**Learning rate increased from 2e-5 to 3e-5:** With a small dataset of 250 examples, the default 2e-5 produced very slow convergence and poor validation accuracy through 6 epochs. Increasing to 3e-5 improved learning speed without causing instability.
 
**Epochs set to 7:** Tested 3, 6, and 7 epochs. At 3 epochs the model underfitted badly. At 6 epochs with the original settings, it collapsed to predicting `hot_take` for everything. At 7 epochs with the adjusted batch size and learning rate, all four classes showed non-zero F1.
 
**Weighted loss attempted and reverted:** A custom `WeightedTrainer` with class weights `[1.0, 2.5, 2.5, 2.5]` was tested to address `hot_take` dominance. It improved minority class recall but hurt `hot_take` F1 to zero in some runs. The standard `Trainer` with adjusted hyperparameters produced more stable results across all classes.
 
---
 
## Baseline Comparison
 
### Approach
 
The zero-shot baseline used Groq's `llama-3.3-70b-versatile` model with the following system prompt:
 
```
You are classifying posts and comments from Indian music communities on Reddit (r/MusicIndia, r/india, r/LetsTalkMusic).
Assign each post to exactly one of the following categories.
 
hot_take: a bold or controversial opinion about an artist, song, or genre stated confidently without supporting evidence or reasoning.
Example: "Mainstream music is totally shit nowadays. It feels like beats are made first and lyrics are forced into it."
 
reaction: an immediate emotional or expressive response to a song, artist, or release, with little to no argument or recommendation.
Example: "Moon Child, my gosh, I could listen to that song all day, every day."
 
recommendation: a post primarily aimed at suggesting music to others, whether a single artist, a list, or a curated playlist.
Example: "If you're into folksy, When Chai Met Toast. If you're into rock-ish stuff, Shine A Light On Me by Raghav Meattle."
 
argument: a post that directly responds to or challenges another person's opinion, agreeing or disagreeing with reasoning.
Example: "Totally agree man, mainstream music is totally shit nowadays."
 
Respond with ONLY the label name. Do not explain your reasoning.
```
 
The baseline was run on the locked test set (38 examples) before any fine-tuning. All 38 responses were parseable.
 
---
 
## Evaluation Report
 
### Results Summary
 
| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | 0.600 |
| Fine-tuned DistilBERT | 0.553 |
 
### Fine-Tuned Model — Per-Class Metrics
 
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| hot_take | 0.47 | 0.54 | 0.50 | 13 |
| reaction | 0.64 | 0.78 | 0.70 | 9 |
| recommendation | 0.57 | 0.44 | 0.50 | 9 |
| argument | 0.60 | 0.43 | 0.50 | 7 |
| **accuracy** | | | **0.553** | **38** |
| macro avg | 0.57 | 0.55 | 0.55 | 38 |
 
### Baseline — Per-Class Metrics
 
| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| hot_take | 0.64 | 0.64 | 0.64 | 11 |
| reaction | 0.43 | 0.86 | 0.57 | 7 |
| recommendation | 1.00 | 0.33 | 0.50 | 6 |
| argument | 1.00 | 0.50 | 0.67 | 6 |
| **accuracy** | | | **0.600** | **30** |
| macro avg | 0.77 | 0.58 | 0.59 | 30 |
 
### Confusion Matrix (Fine-Tuned Model)
 
|  | Predicted: hot_take | Predicted: reaction | Predicted: recommendation | Predicted: argument |
|---|---|---|---|---|
| **True: hot_take** | 7 | 2 | 2 | 2 |
| **True: reaction** | 2 | 7 | 0 | 0 |
| **True: recommendation** | 4 | 1 | 4 | 0 |
| **True: argument** | 2 | 1 | 1 | 3 |
 
### Wrong Predictions Analysis
 
**Wrong prediction #1:**
> Text: "Listening to full albums changed everything for me. You get context for the songs that you just can't get from a playlist."
> True label: `recommendation` — Predicted: `hot_take` (confidence: 0.90)
 
This post is making a suggestion about how to listen to music, which is why it was labeled `recommendation`. However, it reads as a confident opinion statement ("changed everything for me") without naming a specific artist or album to listen to. The model likely picked up on the assertive, opinion-like framing rather than the implicit recommendation embedded in the advice. This points to a boundary weakness: posts that give listening advice without naming specific music sit awkwardly between `hot_take` and `recommendation`.
 
**Wrong prediction #2:**
> Text: "Streaming hasn't ruined music discovery, it has democratized it. Small artists can now reach global audiences without labels."
> True label: `argument` — Predicted: `hot_take` (confidence: 0.79)
 
This is a direct rebuttal of a common claim ("streaming ruined music discovery"), which is why it was labeled `argument`. But without the prior context of the thread, the model sees a confident standalone opinion and classifies it as `hot_take`. This is a systematic weakness: `argument` posts taken out of thread context often look identical to `hot_take` posts. The model has no access to conversational context, so it cannot know whether a statement is responding to someone else.
 
**Wrong prediction #3:**
> Text: "Indie Indian music peaked with Peter Cat Recording Co, everything after is derivative."
> True label: `hot_take` — Predicted: `recommendation` (confidence: 0.90)
 
This is a bold opinion with no evidence, clearly a `hot_take`. The model predicted `recommendation` with high confidence, likely because the post names a specific band (Peter Cat Recording Co) and the model associated named artist mentions with recommendations. This reveals that the model learned surface-level features — artist name presence — rather than the intent of the post.
 
### Sample Classifications
 
| Text | Predicted Label | Confidence | Correct? |
|---|---|---|---|
| "Moon Child, my gosh, I could listen to that song all day, every day." | reaction | 0.91 | ✅ |
| "If you're into folksy, When Chai Met Toast. If you're into rock, Shine A Light On Me by Raghav Meattle." | recommendation | 0.88 | ✅ |
| "AR Rahman hasn't made anything genuinely great since the 2000s." | hot_take | 0.84 | ✅ |
| "I disagree that AR Rahman has declined. Rockstar and Highway both have incredible scores." | argument | 0.72 | ✅ |
| "Indie Indian music peaked with Peter Cat Recording Co, everything after is derivative." | recommendation | 0.90 | ❌ (true: hot_take) |
 
The first correct prediction ("Moon Child, my gosh...") is reasonable because the text contains no artist suggestion, no opinion claim, and no rebuttal — it is purely an emotional expression of enjoyment, which maps clearly onto `reaction`.
 
---
 
## Reflection: What the Model Learned vs. What Was Intended
 
The biggest issue came from the distinction between `hot_take` and `argument`. When labeling, I treated them differently based on intent: a `hot_take` is an unprompted opinion, while an `argument` responds to someone else's claim. However, the model only sees individual posts and comments, not the surrounding thread. Because of that, it couldn't learn the context-based distinction and instead relied on writing style.

In practice, the model seemed to associate strong, confident statements with `hot_take`, while comments containing agreement, disagreement, or softer language were more likely to be classified as `argument` or `reaction`.

The most common error was **predicting `argument` as `hot_take`**. Many comments labeled as arguments looked almost identical to hot takes when viewed on their own. For example, a comment like *"Streaming hasn't ruined music discovery, it has democratized it"* was labeled as an `argument` because it was responding to another person's claim, but without that context it simply looks like a bold opinion. Since the model never sees the original discussion, it had no reliable way to learn this difference.

Another pattern was **artist names leading to `recommendation` predictions**. Some `hot_take` examples that mentioned a specific artist were incorrectly classified as recommendations. This suggests the model learned to associate artist names with recommendation posts instead of focusing on whether the writer was actually recommending something.

Overall, these errors highlight a mismatch between the label definitions and the information available to the model. The `argument` label depends heavily on conversation context, while the model only receives a single post at a time. As a result, some distinctions that were easy for a human annotator were difficult for the classifier to learn.
 
---
 
## Spec Reflection
 
**One way the spec helped:** The requirement to write `planning.md` before collecting data forced me to define the labels and edge-case rules upfront. Having those rules written down made annotation much more consistent, especially for difficult boundaries like `argument` vs `hot_take`. Without that planning step, I likely would have made different labeling decisions as the dataset grew.

**One way implementation diverged:** The project plan assumed that fine-tuning would improve performance over the zero-shot baseline. However, the final results showed the opposite: the fine-tuned model achieved **0.553 accuracy**, while the baseline achieved **0.600**. Instead of treating this as a failed project, I focused on understanding why it happened. The evaluation showed that the main issue was the `argument` label, which depends on conversational context that the model never sees. Investigating dataset size, class balance, and training choices helped explain the result and provided useful insight into the limitations of sentence-level classification.
---
 
## AI Usage
 
**Instance 1 — CSV labeling assistance:** Claude was given batches of unlabeled posts with the label definitions and asked to pre-label them. Every pre-assigned label was reviewed and corrected before finalizing. Around 15% of pre-labels were changed, mostly cases where Claude labeled opinionated posts as `argument` when they were standalone (not replying to anyone) and should have been `hot_take`.
 
**Instance 2 — Failure pattern analysis:** After evaluation, the list of 15 wrong predictions was provided to Claude with a request to identify common patterns. Claude identified the `argument` → `hot_take` confusion and the artist name → `recommendation` pattern. Both were verified by manually re-reading the wrong predictions before being included in the evaluation report.
