"use client";

import TrainingViz from "@/components/TrainingViz";
import Continue from "./Continue";
import { SceneText, SceneTitle } from "./ui";
import type { SceneProps } from "./types";

export default function TrainingScene({ onNext }: SceneProps) {
  return (
    <div className="text-center">
      <SceneTitle>How did it learn all this?</SceneTitle>
      <SceneText>
        Everything so far — the embeddings, the attention, the blocks — started as
        random numbers that meant nothing. Training repeats four steps thousands
        of times: <strong className="text-stone-700">predict</strong>, measure how
        wrong it was (the <strong className="text-stone-700">loss</strong>),
        compute how to fix every parameter (<strong className="text-stone-700">
        backprop</strong>), and nudge them all a hair. Drag the scrubber to watch
        random noise turn into Shakespeare:
      </SceneText>

      <div className="mx-auto mt-6 max-w-2xl text-left">
        <TrainingViz />
      </div>

      <SceneText delay={0.2} className="mt-5 text-[15px]">
        We also watch the loss on text the model never trains on. If it kept
        dropping on the training text but stalled on the unseen text, the model
        would just be <em>memorizing</em> — that&apos;s overfitting. Here both fall
        together, so it&apos;s genuinely learning.
      </SceneText>

      <Continue onClick={onNext} label="Now watch the finished model write" delay={0.3} />
    </div>
  );
}
