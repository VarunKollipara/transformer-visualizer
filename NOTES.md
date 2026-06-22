# NOTES — Plain-Language Concept Log

This is the running glossary for the project. Every new term gets a short entry
that a curious non-expert could follow. It doubles as the raw material for the
website's "How does an AI actually work?" section, so it's written for that
audience, not for specialists.

Entries are added as we encounter each idea, newest concepts at the bottom of
each phase.

---

## Phase 1 — Data & tokenizer

### encode / decode, stoi / itos
The tokenizer is two dictionaries. We scan the whole corpus once to find its
vocabulary — the **sorted** set of unique characters (65 of them for
tinyshakespeare) — and number them 0..64 by sorted position. `stoi`
("string-to-integer") maps each character to its ID and powers **encode**;
`itos` ("integer-to-string") is the reverse and powers **decode**. Sorting is
what makes the numbering a fixed contract: a model is trained against one
specific character↔number mapping, so that mapping must be identical every run,
or the model would read scrambled input. A correct tokenizer is **lossless**:
`decode(encode(text)) == text` exactly. Fun detail you can see in the encoded
IDs — space is ID 1 and newline is ID 0 (the lowest IDs, because sorting is by
Unicode code point), so word boundaries are visible right in the numbers.

### Context window (block_size) and the (B, T) batch
The model never reads the whole corpus at once. We encode the corpus into one
long stream of IDs, then repeatedly grab small **random windows** from it. Two
knobs: **block_size** (a.k.a. context length) is how many tokens the model sees
at once — the width of one window; **batch_size** is how many independent
windows we stack and process in parallel each step (GPUs are fast because they
handle many sequences simultaneously). A batch's input `x` and target `y` are
both shaped **(B, T)** — B = batch_size rows, T = block_size columns — where `y`
is `x` slid one position to the right. A neat consequence: a single window of
length T contains T separate predictions (position t sees tokens 0..t and
predicts t+1), so the model learns from T signals per window, not one. This is
also why attention must later be **causal**: position t may only look at tokens
0..t, never ahead, or it would see the answer it's supposed to predict.

### Train / validation split (overfitting)
Before training we set aside the last ~10% of the stream as a **validation
set** the model never trains on. During training we watch two loss numbers: on
the training data and on this held-out data. If training loss keeps falling but
validation loss flattens or rises, the model is **overfitting** — memorizing the
training text instead of learning the language's patterns. The val set is our
honesty check: it measures generalization to text the model has never seen.

---

## Phase 2 — The model

### Token embedding
A token's integer ID is an arbitrary label — you can't do meaningful math on it
and there's nothing to learn in a frozen integer. So the model's first layer
replaces each ID with a **learnable vector**. The embedding is one matrix of
shape (vocab_size, n_embd): one row per token, each row a vector of length
`n_embd` (the "channels", `C`). Embedding a token is just looking up its row, so
a batch of IDs shaped (B, T) becomes vectors shaped (B, T, C). Every number in
the table is a parameter, started random and tuned by backprop — so the model
learns its own representation of each character, and tokens used similarly drift
toward similar vectors. Two facts to hold: the same ID always maps to the same
vector (it's the same row), and `nn.Embedding` is nothing more than this — a
learnable matrix you index into.

### Positional embedding
Token embeddings know *what* a token is but not *where* it sits — every `'e'`
gets the identical vector regardless of position. That loses order, and order
carries meaning ("dog bites man" vs "man bites dog"). The fix is a second
learnable table, shape (block_size, C), indexed by **position** (0, 1, ...,
block_size-1) instead of token identity. For a token at position t we add
`pos_table[t]` to its word vector — same width C, so it's an element-wise sum.
The combined vector `h = token_emb + pos_emb` (shape (B, T, C)) now encodes both
identity and location, so the same letter at two positions gets two different
vectors. All sequences in a batch share the same position table (position 3 is
"the 4th slot" regardless of content), which is why a (T, C) position tensor
broadcasts cleanly over the (B, T, C) batch. This GPT-style "learned absolute"
scheme is one option; the original Transformer used fixed sine/cosine patterns
instead.

### Self-attention (one head)
The mechanism that lets each token pull in information from earlier tokens.
Core idea in one line: **attention is a weighted average where the weights mean
"how relevant."** Each token produces three vectors via learned matrices: a
**query** ("what I'm looking for"), a **key** ("what I am"), and a **value**
("the info I'll hand over"). To find how relevant token B is to token A, compare
A's query with B's key (a dot product → one relevance score). Steps: (1) make
q, k, v; (2) score every query against every key → a (T, T) grid; (3) **causal
mask** — set future positions to -inf so a token can't look ahead; (4) softmax
each row → percentages that sum to 1 (the attention weights); (5) blend the
values by those weights → each token's new, context-aware vector. The weights
form a lower-triangular matrix (the empty upper-right is the masked future).
"Self" = queries, keys, values all come from the same sequence. We divide scores
by sqrt(head_size) so they don't grow large and make softmax too spiky. Note: in
an untrained model the weights are meaningless noise — the *structure* is real,
but useful attention patterns only emerge after training.

### Logits and the language-model head
After attention gives each token a context-aware vector, one final linear layer
(the "language-model head") maps that vector to **vocab_size raw scores** — one
per possible next character. These raw, un-normalized scores are called
**logits**. Softmax turns them into the probability distribution; bigger logit =
higher probability. So the model's output shape is (B, T, vocab_size): a full
next-character score vector at every position.

### Cross-entropy loss (in code)
The training loss. It is exactly the `-log(probability assigned to the true next
character)` from Phase 0, averaged over all positions. PyTorch's
`F.cross_entropy` takes the **raw logits** (it does the softmax internally) plus
the integer targets, so we never softmax by hand for the loss. It expects shape
(N, vocab) for logits and (N,) for targets, so we flatten the (B, T, vocab)
logits and (B, T) targets into one long list of N = B*T predictions. A brand-new
model scores ~ln(vocab_size) (~4.17 for us); training drives it down.

### Generation / sampling
Turning the model into a text generator = the autoregressive loop: feed the
current text, take the logits at the **last** position, softmax to a probability
distribution, **sample** one character from it (torch.multinomial), append it,
repeat. We crop the input to the last block_size tokens each step because the
positional table only knows that many positions. Sampling (vs. always taking the
single most-likely char) is what gives varied, non-repetitive text.

---

## Phase 3 — Training

### The training loop (optimizer, the four lines)
Training is the Phase 0 loop in code, repeated thousands of times on random
batches. Each step: (1) **forward** — `logits, loss = model(xb, yb)`; (2)
**reset** — `optimizer.zero_grad()` clears the previous step's gradients (PyTorch
adds new gradients onto old ones by default, so we must wipe them); (3)
**backward** — `loss.backward()` backprops, filling in the gradient for every
parameter; (4) **step** — `optimizer.step()` nudges every parameter one step
downhill. Only step (4) actually changes the model; the others compute *how* to
change it. The **optimizer** (we use AdamW) holds all the parameters and applies
the update; the **learning rate** is the step size. We periodically measure loss
on both train and val sets to watch for overfitting.

### First result
A tiny single-head model (~23K params, block_size 32) trained on CPU drove loss
from ~4.18 (the ln(65) baseline) to ~2.34 in 3000 steps, with train and val
falling together (no overfitting). Its samples aren't English but capture the
*format* of a play — capitalized speaker names with colons, line breaks, and
real short words — emerging purely from next-character prediction. Architecture
upgrades (multi-head, MLP, stacked blocks) come next to push quality higher.

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
