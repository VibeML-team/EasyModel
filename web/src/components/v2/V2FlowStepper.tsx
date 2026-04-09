type StepItem = {
  id: number;
  title: string;
};

type V2FlowStepperProps = {
  steps: StepItem[];
  currentStep: number;
};

export function V2FlowStepper({ steps, currentStep }: V2FlowStepperProps) {
  return (
    <nav aria-label="V2 flow steps" className="flow-stepper">
      {steps.map((step) => {
        const isActive = step.id === currentStep;
        const isCompleted = step.id < currentStep;
        const className = isActive
          ? "step-pill step-pill-active"
          : isCompleted
            ? "step-pill step-pill-complete"
            : "step-pill";

        return (
          <div key={step.id} className={className}>
            <span className="step-index">{step.id}</span>
            <span className="step-title">{step.title}</span>
          </div>
        );
      })}
    </nav>
  );
}
