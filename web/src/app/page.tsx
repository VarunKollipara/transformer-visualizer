import ApiStatus from "@/components/ApiStatus";
import AttentionExplorer from "@/components/AttentionExplorer";
import Concept from "@/components/Concept";
import CorpusSample from "@/components/CorpusSample";
import EmbeddingMap from "@/components/EmbeddingMap";
import GenerationDemo from "@/components/GenerationDemo";
import PipelineDiagram from "@/components/PipelineDiagram";
import Section from "@/components/Section";
import SectionNav from "@/components/SectionNav";
import SoftmaxPlayground from "@/components/SoftmaxPlayground";
import TokenizerDemo from "@/components/TokenizerDemo";
import TrainingViz from "@/components/TrainingViz";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-3xl px-5 pb-24">
      <SectionNav />
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
        id="generation"
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
        id="corpus"
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
        id="tokens"
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
        id="embeddings"
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
        id="attention"
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
        id="training"
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
        id="choice"
        title="From scores to a choice"
        subtitle={
          <>
            At each step the model outputs a raw score —{" "}
            <Concept id="logits">logits</Concept> — for every possible next
            character. <Concept id="softmax">Softmax</Concept> turns those into
            probabilities, and <Concept id="temperature">temperature</Concept>{" "}
            controls how boldly it chooses. These are the model&apos;s{" "}
            <em>real</em> scores — drag things and watch the choice reshape:
          </>
        }
      >
        <SoftmaxPlayground />
      </Section>

      <Section
        step={8}
        accent="rose"
        id="pipeline"
        title="Putting it together"
        subtitle={
          <>
            That&apos;s the whole machine, end to end. Click any stage to jump
            back to it:
          </>
        }
      >
        <PipelineDiagram />
        <p className="mt-5 text-[15px] leading-relaxed text-stone-600">
          Scale this up — far more data, billions of{" "}
          <Concept id="parameter">parameters</Concept>, a much longer context —
          and you get ChatGPT. The ideas on this page are exactly the same; only
          the numbers change.
        </p>
      </Section>

      <footer className="mt-8 border-t border-stone-200 pt-8 text-sm text-stone-400">
        Built from scratch — character tokenizer, embeddings, multi-head
        attention, MLP, LayerNorm and residual blocks — in PyTorch, served by
        FastAPI, visualized with Next.js.
      </footer>
    </main>
  );
}
