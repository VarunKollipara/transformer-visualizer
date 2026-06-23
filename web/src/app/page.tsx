import ApiStatus from "@/components/ApiStatus";
import Aside from "@/components/Aside";
import AttentionExplorer from "@/components/AttentionExplorer";
import Concept from "@/components/Concept";
import CorpusSample from "@/components/CorpusSample";
import EmbeddingMap from "@/components/EmbeddingMap";
import GenerationDemo from "@/components/GenerationDemo";
import GoDeeper from "@/components/GoDeeper";
import PipelineDiagram from "@/components/PipelineDiagram";
import QKVDiagram from "@/components/QKVDiagram";
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
        <Aside>
          it&apos;s like extreme autocomplete. The model has no plan for the
          sentence — it just guesses the next single character that feels likely,
          adds it, then re-reads everything and guesses again.
        </Aside>
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
        <Aside>
          imagine a person who has only ever read Shakespeare — no school, no
          internet, nothing else. They&apos;d soak up his words and rhythm, but
          couldn&apos;t tell you today&apos;s date. That&apos;s exactly our model.
        </Aside>
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
        <Aside>
          computers can&apos;t &ldquo;see&rdquo; letters, only numbers. So we give
          every character a numbered locker — <span className="font-mono">a</span>
          →39, <span className="font-mono">b</span>→40 — and from here on the
          model only ever deals with locker numbers.
        </Aside>
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
        <Aside>
          a locker number says nothing about what&apos;s inside. So each character
          also gets a list of traits — numbers the model makes up during training
          — that place similar characters near each other, like seating party
          guests by how alike they are.
        </Aside>
        <GoDeeper title="Why 128 dimensions — and what the 2D map hides">
          <p>
            Each token&apos;s embedding is a vector of 128 numbers, not 2. More
            dimensions give the model more independent &ldquo;axes&rdquo; to
            express how tokens differ and relate. We can&apos;t draw 128
            dimensions, so the map above uses <strong>PCA</strong> to flatten
            them down to the two directions that vary the most. That reveals
            structure but loses detail — two points can look close here yet
            differ along an axis we dropped.
          </p>
          <p className="mt-2">
            The numbers start random and are tuned during training, so any
            structure you see (uppercase apart from lowercase, punctuation
            clustering) was <em>learned</em> purely from predicting the next
            character.
          </p>
        </GoDeeper>
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
        <Aside>
          to guess the next letter, each position &ldquo;asks around&rdquo;: which
          earlier letters matter to me right now? It can only ask the ones before
          it (no peeking at the answer), then blends their hints together.
        </Aside>
        <GoDeeper title="How attention is computed (query, key, value)">
          <QKVDiagram />
          <p>
            Each token is turned into three vectors by learned matrices: a{" "}
            <strong>query</strong> (what it&apos;s looking for), a{" "}
            <strong>key</strong> (what it offers), and a{" "}
            <strong>value</strong> (the info it passes on). The relevance of token{" "}
            <em>j</em> to token <em>i</em> is the dot product of i&apos;s query
            and j&apos;s key. Those scores are divided by √(head size) to stay
            stable, the future is masked to −∞, then softmax turns them into
            weights that sum to 1 — and the output is the weighted sum of the
            values.
          </p>
          <p className="mt-2">
            A &ldquo;<strong>head</strong>&rdquo; is one such query/key/value set.
            Running several in parallel (4 here) lets the model track different
            relationships at once; their outputs are concatenated and mixed.
            Stack a few of these attention-plus-MLP blocks and you have a
            transformer.
          </p>
        </GoDeeper>
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
        <Aside>
          at first the model scribbles nonsense. Each round it&apos;s told how
          wrong it was and tweaks its ~619,000 dials a hair in the right
          direction. Repeat thousands of times and the scribbles turn into
          sentences.
        </Aside>
        <GoDeeper title="The four steps of one training iteration">
          <ol className="list-decimal space-y-1 pl-5">
            <li>
              <strong>Forward</strong> — run a batch of text through the model and
              measure the <em>loss</em>: the negative log of the probability it
              gave the true next character.
            </li>
            <li>
              <strong>Backward (backprop)</strong> — compute, for every one of the
              ~619,000 parameters, which way to nudge it to lower the loss.
            </li>
            <li>
              <strong>Step</strong> — the optimizer moves every parameter a tiny
              amount in that direction.
            </li>
            <li>
              <strong>Repeat</strong> — thousands of times, on fresh batches.
            </li>
          </ol>
          <p className="mt-2">
            We also track the loss on held-out <strong>validation</strong> text
            the model never trains on. If training loss keeps dropping while
            validation loss stalls, the model is memorizing rather than learning
            — <em>overfitting</em>. Here they fall together, which is what we
            want.
          </p>
        </GoDeeper>
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
        <Aside>
          think of a spin-the-wheel where each possible next letter gets a slice.
          Bigger score = bigger slice. <em>Temperature</em> makes the wheel fairer
          (more random) or more rigged toward the current favorite.
        </Aside>
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
