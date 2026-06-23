// Plain-language definitions for the "click any concept to learn more" feature.
// Sourced from the project's NOTES.md — written for a curious non-expert.

export type GlossaryEntry = { title: string; body: string };

export const glossary: Record<string, GlossaryEntry> = {
  token: {
    title: "Token",
    body: "The smallest unit of text the model reads and writes. We use one character per token, so 'H', ' ' (space) and even newline are each one token. The model never sees letters — only the integer ID of each token.",
  },
  vocabulary: {
    title: "Vocabulary",
    body: "The complete list of every token the model knows — the sorted set of unique characters in the training text (65 of them for our Shakespeare corpus). A token's ID is just its position in this sorted list.",
  },
  "next-token": {
    title: "Next-token prediction",
    body: "The model's entire job: given the text so far, output a probability for every possible next character. Generating text means doing this over and over, each time adding the predicted character and predicting again.",
  },
  embedding: {
    title: "Embedding",
    body: "A learnable vector (a list of numbers) standing in for each token. Bare integer IDs carry no meaning, so the model converts each ID into a vector it can do math on and tune during training. Similar tokens drift toward similar vectors.",
  },
  position: {
    title: "Positional embedding",
    body: "Token embeddings know WHAT a token is but not WHERE it sits. A second learnable vector per position is added in, so the same letter at different spots gets different vectors — letting the model use word order.",
  },
  attention: {
    title: "Self-attention",
    body: "The mechanism that lets each token look back at earlier tokens and pull in the relevant ones. It scores how relevant every earlier token is, turns those into percentages that sum to 1, and blends their information accordingly.",
  },
  "causal-mask": {
    title: "Causal mask",
    body: "A token may only attend to itself and earlier tokens, never future ones — because when generating, the future doesn't exist yet. This is why the attention grid is a lower triangle: the upper-right (the future) is blocked.",
  },
  head: {
    title: "Attention head",
    body: "One independent attention pattern. Running several 'heads' in parallel lets the model track different relationships at once — one head might follow the previous letter, another the start of the word — then combine them.",
  },
  logits: {
    title: "Logits",
    body: "The raw, un-normalized scores the model outputs for each possible next character — one number per vocabulary token. Bigger logit, higher chance. Softmax turns them into actual probabilities.",
  },
  softmax: {
    title: "Softmax",
    body: "A function that turns a list of raw scores (logits) into probabilities: all positive and summing to 1. It exponentiates each score then divides by the total, so bigger scores get exponentially more probability.",
  },
  temperature: {
    title: "Temperature",
    body: "A dial on randomness during generation. Low temperature (<1) sharpens the distribution toward the single most likely character (safe, repetitive). High temperature (>1) flattens it (more varied, more mistakes). 1.0 is the model's raw distribution.",
  },
  loss: {
    title: "Loss (cross-entropy)",
    body: "A single number measuring how wrong the model is: the negative log of the probability it gave the TRUE next character, averaged over all positions. Training pushes it down. A brand-new model scores about 4.17; ours reaches ~1.66.",
  },
  training: {
    title: "Training",
    body: "Repeating four steps thousands of times: predict (forward), measure wrongness (loss), compute how to improve every parameter (backprop), and nudge them downhill (optimizer step). The model's 'knowledge' is just the final settings of its parameters.",
  },
};
