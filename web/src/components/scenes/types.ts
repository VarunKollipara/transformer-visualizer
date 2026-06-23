// Every scene receives the same navigation handlers; it uses whichever it needs.
export type SceneProps = {
  onNext: () => void;
  onBack: () => void;
};
