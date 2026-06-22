# NOTES — Plain-Language Concept Log

This is the running glossary for the project. Every new term gets a short entry
that a curious non-expert could follow. It doubles as the raw material for the
website's "How does an AI actually work?" section, so it's written for that
audience, not for specialists.

Entries are added as we encounter each idea, newest concepts at the bottom of
each phase.

---

## Phase 0 — Foundations

### Language model
A **language model** is a function that, given some text, predicts what text is
likely to come next. That's the whole job. A modern "AI chatbot" is a very large
language model wrapped in some extra machinery — but at its heart it is just
guessing the next bit of text, over and over. Ours will guess one character at a
time.

### Token
A **token** is the smallest unit of text the model reads and writes — its
"atom" of language. You get to decide what a token is. It could be a word, a
piece of a word, or (in our project) a single character. The model never sees
letters or words directly; it only ever sees tokens, and only ever as numbers.

### Vocabulary
The **vocabulary** is the complete, fixed list of every token the model knows.
For us it's the sorted set of unique characters that appear in our training
text — roughly 65 symbols for English (letters, digits, punctuation, space,
newline). Each token is assigned an integer ID by its position in this list, so
the mapping between characters and numbers is fixed and reproducible. "Vocab
size" is just how many distinct tokens exist; it sets the length of every
prediction the model makes.

### Character-level tokenization
Choosing **one character = one token**. Encoding text means replacing each
character with its vocabulary ID (e.g. with vocab `' ',d,e,h,l,o,r,w` →
`0..7`, the word `"hello"` becomes `[3, 2, 4, 4, 5]`). Decoding runs the table
backwards to turn IDs back into characters. We use character-level because the
tokenizer becomes trivial and transparent — no clever sub-word algorithm to
hide what's going on — while every interesting transformer idea still appears
unchanged. The cost is that the model has to work harder to spell, since it
builds words one letter at a time.

### Next-token prediction (the training task)
The model's actual task: given a sequence of tokens, output a probability for
**every** token in the vocabulary as the candidate for the *single next* token.
We turn ordinary text into training examples by shifting it by one position —
the target for each position is simply the character that actually follows it.
For the text `"hello"`: inputs `"hell"` → targets `"ello"`, read as "after `h`
predict `e`, after `he` predict `l`, ..." Because the next character *is* the
answer, the data labels itself — no human annotation needed. This is what
"self-supervised" means.

### Autoregressive generation
How the model writes new text at inference time: predict a probability
distribution for the next token, pick one token from it, **append it to the
input**, and repeat — feeding the model its own output. Each pass produces
exactly one token. The model has no memory between steps and no plan for the
future; it re-reads the whole text-so-far each time and makes coherence emerge
from each next-token guess being conditioned on everything before it.
("Autoregressive" = it regresses on / depends on its own previous outputs.)

---
