/**
 * Presentational components for the chat. They render result fields only; schedules, costs,
 * and labels are never computed here (ui-contract.md §5).
 */
import { useId } from "react";

import { formatMoney, formatTime, estimateLabelText, slotName, totalCostText } from "./format.js";
import { itinerarySteps } from "./itinerary.js";

/** The chat messages, then the pending user message while its turn runs. */
export function Conversation({ messages, pending }) {
  const shown = pending === null ? messages : [...messages, { role: "user", text: pending }];
  return (
    <div role="log" aria-label="Conversation" className="conversation">
      {shown.map((message, index) => (
        <p key={index} data-testid="message" data-role={message.role} className="message">
          {message.text}
        </p>
      ))}
    </div>
  );
}

function Step({ step, timeZone }) {
  if (step.kind === "leg") {
    return (
      <>
        {step.mode} {step.minutes} min
        {step.estimate !== null && <span className="estimate"> {step.estimate}</span>}
      </>
    );
  }
  return (
    <>
      <strong>{step.label}</strong> {step.verb}{" "}
      <time dateTime={step.time}>{formatTime(step.time, timeZone)}</time>
    </>
  );
}

function Option({ option, request, estimateLabels }) {
  const headingId = useId();
  const { itinerary } = option;
  const slot = option.labels[0];
  const steps = itinerarySteps(
    itinerary,
    request.gap.start_location.label,
    request.commitment.location.label,
    estimateLabels,
  );
  return (
    <article aria-labelledby={headingId} className="option">
      <h2 id={headingId}>{slotName(slot)}</h2>
      {(slot === "cheapest" || slot === "best_travel") && <p>Among the remaining plans</p>}
      {option.recommendation_reason !== null && <p>{option.recommendation_reason}</p>}
      <ol aria-label="Itinerary">
        {steps.map((step, index) => (
          <li key={index}>
            <Step step={step} timeZone={request.commitment.timezone} />
          </li>
        ))}
      </ol>
      <p data-testid="total-cost">
        {totalCostText(itinerary.total_cost, itinerary.total_includes_estimates)}
      </p>
      <ul aria-label="Costs">
        {itinerary.candidate.costs.map((cost, index) => {
          const label = estimateLabelText(cost.evidence_ids, estimateLabels);
          return (
            <li key={index}>
              {cost.description}: {formatMoney(cost.amount)}
              {label !== null && <span className="estimate"> {label}</span>}
            </li>
          );
        })}
      </ul>
      {itinerary.weather_warnings.map((warning, index) => (
        <p key={index} data-testid="weather-warning" className="warning">
          {warning}
        </p>
      ))}
    </article>
  );
}

/** The questions, retry question, or options of the last result; nothing for other kinds. */
export function LastResult({ lastResult, request }) {
  const result = lastResult?.result;
  switch (result?.kind) {
    case "clarification":
      return (
        <ul aria-label="Questions to answer">
          {result.questions.map((question, index) => (
            <li key={index}>{question.question}</li>
          ))}
        </ul>
      );
    case "no_feasible_result":
      return <p data-testid="retry-question">{result.retry_question}</p>;
    case "validated_options":
      return (
        <section aria-label="Itinerary options" className="options">
          {result.options.map((option, index) => (
            <Option
              key={index}
              option={option}
              request={request}
              estimateLabels={lastResult.estimate_labels}
            />
          ))}
        </section>
      );
    default:
      return null;
  }
}
