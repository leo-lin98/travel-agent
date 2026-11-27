/**
 * Pure checks that a result has every field the UI renders, with its contract type
 * (ui-contract.md §6, §7; TR1, TR2). They check only what is displayed; the backend stays
 * the authority on everything else in the continuation.
 */
import { modeName, slotName } from "./format.js";
import { isObject } from "./json.js";

const isText = (value) => typeof value === "string";
const isList = (value, isItem) => Array.isArray(value) && value.every(isItem);
const isIds = (value) => isList(value, isText);
const isTime = (value) => isText(value) && !Number.isNaN(Date.parse(value));
const isMoney = (value) => isObject(value) && isText(value.amount) && isText(value.currency);
const hasLabel = (value) => isObject(value) && isText(value.label);

function isTimeZone(value) {
  if (!isText(value)) {
    return false;
  }
  try {
    new Intl.DateTimeFormat("en-GB", { timeZone: value });
    return true;
  } catch {
    return false;
  }
}

const isStop = (stop) => hasLabel(stop);
const isLeg = (leg) =>
  isObject(leg) &&
  isText(modeName(leg.mode)) &&
  Number.isFinite(leg.duration_minutes) &&
  isIds(leg.evidence_ids);
const isCost = (cost) =>
  isObject(cost) && isText(cost.description) && isMoney(cost.amount) && isIds(cost.evidence_ids);

function isItinerary(itinerary) {
  if (!isObject(itinerary) || !isObject(itinerary.candidate)) {
    return false;
  }
  const { stops, legs, costs } = itinerary.candidate;
  return (
    isList(stops, isStop) &&
    stops.length > 0 &&
    isList(legs, isLeg) &&
    legs.length === stops.length + 1 &&
    isList(costs, isCost) &&
    isList(itinerary.leave_by, isTime) &&
    itinerary.leave_by.length === stops.length + 1 &&
    isTime(itinerary.arrival_at_commitment) &&
    isMoney(itinerary.total_cost) &&
    typeof itinerary.total_includes_estimates === "boolean" &&
    isList(itinerary.weather_warnings, isText)
  );
}

const isOption = (option) =>
  isObject(option) &&
  Array.isArray(option.labels) &&
  isText(slotName(option.labels[0])) &&
  (option.recommendation_reason === null || isText(option.recommendation_reason)) &&
  isItinerary(option.itinerary);

/** Whether the request has the labels and timezone that options are rendered with. */
const isOptionsRequest = (request) =>
  isObject(request) &&
  isObject(request.gap) &&
  hasLabel(request.gap.start_location) &&
  isObject(request.commitment) &&
  hasLabel(request.commitment.location) &&
  isTimeZone(request.commitment.timezone);

/** Whether `result` can be rendered with `estimateLabels` and the continuation's `request`. */
export function isRenderableResult(result, estimateLabels, request) {
  if (!isObject(result)) {
    return false;
  }
  switch (result.kind) {
    case "clarification":
      return isList(
        result.questions,
        (question) => isObject(question) && isText(question.question),
      );
    case "no_feasible_result":
      return isText(result.retry_question);
    case "execution_failure":
      return true;
    case "validated_options":
      return (
        isList(result.options, isOption) &&
        result.options.length > 0 &&
        isObject(estimateLabels) &&
        isOptionsRequest(request)
      );
    default:
      return false;
  }
}

/** Whether a result event has a continuation and a result that can be rendered with it. */
export const isResultEvent = (event) =>
  isObject(event.continuation) &&
  isRenderableResult(event.result, event.estimate_labels, event.continuation.request);
