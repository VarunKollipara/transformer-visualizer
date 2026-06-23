import ApiStatus from "@/components/ApiStatus";
import AttentionExplorer from "@/components/AttentionExplorer";
import Concept from "@/components/Concept";
import GenerationDemo from "@/components/GenerationDemo";
import Section from "@/components/Section";
import TokenizerDemo from "@/components/TokenizerDemo";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-3xl px-5 pb-24">
      {/* Hero */}
      <header className="py-16">
        <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-emerald-400">
          How an AI actually works
        </p>
        <h1 className="text-4xl font-bold leading-tight tracking-tight text-zinc-50 sm:text-5xl">
          A language model, from training data to its final words.
        </h1>
        <p className="mt-5 text-lg leading-relaxed text-zinc-400">
          Everything below is powered by a small{" "}
          <Concept id="next-token">GPT-style model</Concept> built and trained
          from scratch on Shakespeare. No mocks — you&apos;re poking at the real
          thing. Anything in{" "}
          <span className="font-medium text-emerald-400">green</span> is
          clickable to learn more.
        </p>
        <div className="mt-6">
          <ApiStatus />
        </div>
      </header>

      <Section
        step={1}
        title="It only does one thing"
        subtitle={
          <>
            A language model is a function for{" "}
            <Concept id="next-token">next-character prediction</Concept>: given
            the text so far, it outputs a probability for every possible next
            character. Writing text is just doing that over and over.
          </>
        }
      >
        <GenerationDemo />
      </Section>

      <Section
        step={2}
        title="Text becomes numbers"
        subtitle={
          <>
            A neural network only does math on numbers, so first every character
            is turned into a <Concept id="token">token</Concept> ID using a fixed{" "}
            <Concept id="vocabulary">vocabulary</Concept>. Type below and watch.
          </>
        }
      >
        <TokenizerDemo />
      </Section>

      <Section
        step={3}
        title="Each token looks at the others"
        subtitle={
          <>
            The core of the model is{" "}
            <Concept id="attention">self-attention</Concept>: every token looks
            back at earlier tokens and pulls in the relevant ones. Each{" "}
            <Concept id="head">head</Concept> learns a different pattern, and the{" "}
            <Concept id="causal-mask">causal mask</Concept> stops it peeking
            ahead — that&apos;s why the grid is a triangle. Brighter = more
            attention. Click a row to see that token&apos;s prediction.
          </>
        }
      >
        <AttentionExplorer />
      </Section>

      <Section
        step={4}
        title="…and the loop repeats"
        subtitle={
          <>
            Those predictions are <Concept id="logits">logits</Concept>, turned
            into probabilities by <Concept id="softmax">softmax</Concept>. Pick
            one (the <Concept id="temperature">temperature</Concept> controls how
            adventurously), append it, and feed everything back in. That loop —
            run thousands of times during{" "}
            <Concept id="training">training</Concept> to drive the{" "}
            <Concept id="loss">loss</Concept> down — is the whole story.
          </>
        }
      >
        <p className="text-zinc-400">
          Scroll back up to{" "}
          <span className="font-medium text-zinc-200">Step 1</span> and try a low
          vs. high temperature — you&apos;re watching the exact same loop that
          powers ChatGPT, just at a tiny scale.
        </p>
      </Section>

      <footer className="border-t border-zinc-800 pt-8 text-sm text-zinc-500">
        Built from scratch — character tokenizer, embeddings, multi-head
        attention, MLP, LayerNorm and residual blocks — in PyTorch, served by
        FastAPI, visualized with Next.js.
      </footer>
    </main>
  );
}
