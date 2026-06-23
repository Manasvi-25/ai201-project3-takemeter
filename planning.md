# TakeMeter — Planning Document

## Project Overview

TakeMeter is a fine-tuned text classifier designed to identify different types of music discussions in Indian Reddit communities. Given a post or comment from r/MusicIndia or music-related threads on r/india, it predicts whether the content is a hot take, reaction, recommendation, or argument. These categories capture some of the most common ways people engage in music conversations online.

The idea behind the project is to see whether a model can learn the subtle differences between these forms of discourse that regular community members often recognize instinctively. By classifying posts into these categories, TakeMeter also provides a way to explore what kinds of conversations are most common in Indian music spaces and how people express opinions, recommendations, and disagreements around music.

---
## Community
 
I chose r/MusicIndia and music-related threads from r/india because the conversations there are naturally diverse. The same artist, song, or genre can spark very different kinds of responses: someone recommending music, someone sharing a personal or emotional reaction, someone posting a strong or controversial opinion, or someone disagreeing with another user's take. That range of discussion makes these communities a great fit for a classification task.

What makes them even more interesting is the unique nature of music discussions in Indian online spaces. Posts often mix languages, compare Bollywood and indie music, bring up regional artists and genres, or debate trends in the industry. As a result, the data contains a wide variety of tones and intentions that align well with the classification labels used in this project.

---
 
## Label Taxonomy
 
**`hot_take`** — a bold or controversial opinion about an artist, song, or genre stated confidently without supporting evidence or reasoning.
 
Example 1:
> "I know she has a strong lineage and following, but I can't enjoy her singing. There seems to be a lack of control and a certain kind of fake depth in her voice."
 
Example 2:
> "Mainstream music is totally shit nowadays. It feels like beats are made first and lyrics are forced into it."
 
---
 
**`reaction`** — an immediate emotional or expressive response to a song, artist, or release, with little to no argument or recommendation.
 
Example 1:
> "Moon Child, my gosh, I could listen to that song all day, every day."
 
Example 2:
> "Woah tyty tamil and Hindi lyrics brings titli to mind haha"
 
---
 
**`recommendation`** — a post primarily aimed at suggesting music to others, whether a single artist, a list, or a curated playlist, with or without brief reasoning.
 
Example 1:
> "Tamanna by Yawar Abdal. 3 languages all beautifully woven together. Bore Bore from Bluffmaster. Rait Zara Si from Atrangi Re — the Tamil part hits differently."
 
Example 2:
> "If you're into folksy, When Chai Met Toast. If you're into rock-ish stuff, Shine A Light On Me by Raghav Meattle. If you're into dreamy — Moon Child, the F16s."
 
---
 
**`argument`** — a post that directly responds to or challenges another person's opinion, agreeing or disagreeing with reasoning.
 
Example 1:
> "Totally agree man, mainstream music is totally shit nowadays. It feels like beats are first made and lyrics are made to fit forcibly into it. It sucks."
 
Example 2:
> "Anuv Jain not so good" (direct pushback on a prior recommendation in thread)
 
---

## Hard Edge Cases

Some posts can blur the line between a recommendation and a strong opinion. For example, "You need to listen to Mk.gee, most underrated artist out right now" contains both a suggestion and a judgment about the artist.

Decision rule: If the main purpose of the post is to point people toward a song, artist, or album, label it as recommendation. If the focus is on making a bold, debatable claim about the quality, popularity, or status of an artist or genre, label it as hot_take. In general, posts centered on "listen to X" are recommendations, while posts centered on "X is overrated/underrated/the best" are hot takes, even if they also indirectly recommend the artist.

Reaction vs Argument:
Another common overlap occurs when someone responds emotionally to a topic while also engaging with another person's opinion. For example, "Totally agree man, mainstream music is shit" could be interpreted as either a reaction or an argument.

Decision rule: If the comment is directly engaging with a specific claim made by another user, whether by agreeing or disagreeing, label it as argument. If it mainly expresses an emotion, feeling, or personal response without addressing a particular claim, label it as reaction. The key distinction is whether the post is participating in a discussion or simply expressing a response.

## Data Collection Plan

- Source: r/MusicIndia and music-related threads from r/india
- Manually collect public posts and comments
- Target: 220 examples total, about 55 per label
- Store data in a CSV with `text`, `label`, and `notes` columns
- Use `notes` for anything that was difficult or unclear to label
- If a label has too few examples, collect more from threads where that type naturally appears (debates for `argument`, recommendation threads for `recommendation`, etc.)
- Keep the dataset fairly balanced and avoid any single label dominating the dataset

---

## Evaluation Metrics

I will use:

- Overall accuracy
- Per-class F1 score
- Confusion matrix

Accuracy gives a general idea of performance, but it doesn't show whether the model is learning all four labels equally well. Per-class F1 scores help measure how well the model handles each discourse type individually.

The confusion matrix will help identify which labels the model mixes up most often, especially around edge cases like `hot_take` vs `recommendation` or `reaction` vs `argument`.

---

## Definition of Success

The project will be considered successful if:

- Overall accuracy is at least **70%**
- No class has an F1 score below **0.55**
- The fine-tuned model performs better than the zero-shot Groq baseline

If the model beats the baseline but struggles with one specific class, I'll treat that as a partial success and discuss where the classification boundary was difficult to learn.

---

## AI Tool Plan

### Label Stress Testing

- Give Claude the label definitions and edge-case rules
- Ask it to generate 8-10 borderline examples
- Use those examples to check whether the labeling rules are clear and consistent
- Update definitions if repeated ambiguities appear

### Annotation Assistance

- Use Claude to pre-label small batches of examples
- Review every label manually before adding it to the dataset
- Track AI-assisted examples with a `pre_labeled` column (`yes` / `no`)
- Document this process in the README

### Failure Analysis

- After evaluation, provide misclassified examples to Claude
- Ask it to identify possible patterns such as sarcasm, short comments, ambiguous wording, or repeated label confusion
- Verify any suggested patterns manually before including them in the final report
