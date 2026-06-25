// Every scene receives the same navigation handlers; it uses whichever it needs.
// `phase` is the current slide within the scene, controlled by the Experience
// (so the progress bar can address each slide and jumping works). Multi-phase
// scenes cast it to their own Phase union; single-slide scenes ignore it.
export type SceneProps = {
  phase: string;
  onNext: () => void;
  onBack: () => void;
  restart: () => void;
};
