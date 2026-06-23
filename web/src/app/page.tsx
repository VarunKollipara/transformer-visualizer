import ApiStatus from "@/components/ApiStatus";
import AttentionExplorer from "@/components/AttentionExplorer";
import Concept from "@/components/Concept";
import CorpusSample from "@/components/CorpusSample";
import EmbeddingMap from "@/components/EmbeddingMap";
import GenerationDemo from "@/components/GenerationDemo";
import Section from "@/components/Section";
import TokenizerDemo from "@/components/TokenizerDemo";
import TrainingViz from "@/components/TrainingViz";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-3xl px-5 pb-24">
      {/* Hero */}
      <header className="py-16">
        <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-indigo-600">
          How an AI actually works
        </p>
        <h1 className="font-display text-4xl font-semibold leading-[1.1] tracking-tight text-stone-900 sm:text-5xl">
          A language model, from training data to its final words.
        </h1>
        <p className="mt-5 text-lg leading-relaxed text-stone-600">
          Below is a small <Concept id="next-token">GPT-style model</Concept> —
          the same kind of system behind ChatGPT — built and trained from
          scratch on Shakespeare. Every visualization is driven by the{" "}
          <strong className="font-semibold text-stone-800">real model</strong>,
          not a mockup. Anything{" "}
          <span className="font-medium text-indigo-700 underline decoration-indigo-300 decoration-2 underline-offset-2">
            underlined
          </span>{" "}
          is clickable to learn more.
        </p>
        <div className="mt-6">
          <ApiStatus />
        </div>
      </header>

      <Section
        step={1}
        accent="indigo"
        title="It only does one thing"
        subtitle={
          <>
            A language model is a machine for{" "}
            <Concept id="next-token">next-character prediction</Concept>: given
            the text so far, it guesses the next character. Generating writing is
            just doing that over and over — predict a character, add it, predict
            again. Try it:
          </>
        }
      >
        <GenerationDemo />
      </Section>

      <Section
        step={2}
        accent="amber"
        title="What it learned from"
        subtitle={
          <>
            The model has no built-in knowledge. Everything it can do comes from
            one pile of text it was shown during{" "}
            <Concept id="training">training</Concept> — for us, the complete works
            of Shakespeare:
          </>
        }
      >
        <CorpusSample />
      </Section>

      <Section
        step={3}
        accent="sky"
        title="Turning text into numbers"
        subtitle={
          <>
            Computers do math, not letters. So each character becomes a{" "}
            <Concept id="token">token</Concept> — a number — using a fixed{" "}
            <Concept id="vocabulary">vocabulary</Concept> of the 65 characters in
            the text. Type anything and watch it convert:
          </>
        }
      >
        <TokenizerDemo />
      </Section>

      <Section
        step={4}
        accent="violet"
        title="Giving each token meaning"
        subtitle={
          <>
            A bare ID like <span className="font-mono">42</span> means nothing. So
            the model turns each token into an{" "}
            <Concept id="embedding">embedding</Concept> — a{" "}
            <Concept id="vector">vector</Concept> of 128 numbers it{" "}
            <Concept id="training">learns</Concept>. Here are all 65 characters
            placed by their learned vectors (squashed to 2D); ones the model
            treats similarly sit near each other:
          </>
        }
      >
        <EmbeddingMap />
      </Section>

      <Section
        step={5}
        accent="teal"
        title="Letting tokens look at each other"
        subtitle={
          <>
            The key idea behind transformers is{" "}
            <Concept id="attention">self-attention</Concept>: each character
            looks back at earlier ones and pulls in what&apos;s relevant. Several{" "}
            <Concept id="head">heads</Concept> track different patterns, and a{" "}
            <Concept id="causal-mask">causal mask</Concept> blocks peeking ahead
            (that&apos;s why it&apos;s a triangle). Brighter = more attention;
            click a row to see that character&apos;s prediction:
          </>
        }
      >
        <AttentionExplorer />
      </Section>

      <Section
        step={6}
        accent="rose"
        title="How it learned: watch the loss fall"
        subtitle={
          <>
            Training nudges ~619,000{" "}
            <Concept id="parameter">parameters</Concept> to lower the{" "}
            <Concept id="loss">loss</Concept> — a score of how wrong its guesses
            are. Drag the scrubber to <em>watch the same model</em> go from random
            noise to Shakespeare as the loss drops. Train and validation falling
            together means it&apos;s learning, not just{" "}
            <Concept id="overfitting">memorizing</Concept>:
          </>
        }
      >
        <TrainingViz />
      </Section>

      <Section
        step={7}
        accent="indigo"
        title="Putting it together"
        subtitle={
          <>
            That&apos;s the whole machine: text → tokens → vectors → attention
            blocks → a guess for the next character. The model&apos;s raw scores (
            <Concept id="logits">logits</Concept>) become probabilities via{" "}
            <Concept id="softmax">softmax</Concept>; one is picked (how boldly is
            set by <Concept id="temperature">temperature</Concept>), appended, and
            the loop runs again.
          </>
        }
      >
        <div className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
          <ol className="space-y-2 text-[15px] text-stone-700">
            {[
              "Read the text so far as token IDs",
              "Turn each token into a learned vector (+ its position)",
              "Attention + MLP blocks mix in context",
              "Produce a probability for every possible next character",
              "Sample one character, append it, and repeat",
            ].map((s, i) => (
              <li key={i} className="flex gap-3">
                <span className="font-mono font-semibold text-indigo-600">
                  {i + 1}.
                </span>
                {s}
              </li>
            ))}
          </ol>
          <p className="mt-4 text-sm text-stone-500">
            Scale this up — more data, more parameters, bigger context — and you
            get ChatGPT. The ideas on this page are the same; only the numbers
            change.
          </p>
        </div>
      </Section>

      <footer className="mt-8 border-t border-stone-200 pt-8 text-sm text-stone-400">
        Built from scratch — character tokenizer, embeddings, multi-head
        attention, MLP, LayerNorm and residual blocks — in PyTorch, served by
        FastAPI, visualized with Next.js.
      </footer>
    </main>
  );
}
