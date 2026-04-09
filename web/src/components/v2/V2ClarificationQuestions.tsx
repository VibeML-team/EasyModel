import type { V2ClarificationQuestion } from "../../types";

type V2ClarificationQuestionsProps = {
  questions: V2ClarificationQuestion[];
  answers: Record<string, string>;
  onAnswerChange: (questionID: string, value: string) => void;
};

export function V2ClarificationQuestions({
  questions,
  answers,
  onAnswerChange,
}: V2ClarificationQuestionsProps) {
  if (questions.length === 0) {
    return null;
  }

  return (
    <section className="subcard stack-md" aria-label="Clarification questions">
      <h3>Clarification</h3>
      <p className="muted">Please answer these questions so the system can continue with better intent understanding.</p>

      {questions.map((question, index) => (
        <article key={question.id} className="clarify-question stack-sm">
          <h4>
            Q{index + 1}. {question.text}
          </h4>
          {question.context ? <p className="muted">{question.context}</p> : null}

          {question.type === "multiple_choice" && question.options?.length ? (
            <div className="choice-group">
              {question.options.map((option) => {
                const checked = answers[question.id] === option;
                return (
                  <label key={`${question.id}-${option}`} className={checked ? "choice-item choice-item-selected" : "choice-item"}>
                    <input
                      type="radio"
                      name={question.id}
                      value={option}
                      checked={checked}
                      onChange={(event) => onAnswerChange(question.id, event.target.value)}
                    />
                    <span>{option}</span>
                  </label>
                );
              })}
            </div>
          ) : (
            <textarea
              className="field"
              rows={3}
              value={answers[question.id] || ""}
              onChange={(event) => onAnswerChange(question.id, event.target.value)}
              placeholder="Type your answer"
            />
          )}
        </article>
      ))}
    </section>
  );
}
