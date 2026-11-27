import { describe, expect, test } from "vitest";

import { isRenderableResult, isResultEvent } from "./validate.js";

const REQUEST = {
  gap: { start_time: null, start_location: { label: "Station" } },
  commitment: { location: { label: "Dinner" }, time: null, timezone: "Asia/Tokyo" },
};

const OPTIONS = {
  kind: "validated_options",
  message: "Here are your options.",
  options: [
    {
      itinerary: {
        candidate: {
          candidate_id: "c",
          stops: [{ label: "Garden", evidence_ids: [] }],
          legs: [
            { mode: "walking", duration_minutes: 10, evidence_ids: [] },
            { mode: "taxi_rideshare", duration_minutes: 5, evidence_ids: ["ev-taxi"] },
          ],
          costs: [
            {
              description: "Taxi",
              amount: { amount: "900", currency: "JPY" },
              evidence_ids: ["ev-taxi"],
            },
          ],
        },
        leave_by: ["2026-11-20T15:00:00+09:00", "2026-11-20T16:00:00+09:00"],
        arrival_at_commitment: "2026-11-20T16:05:00+09:00",
        total_cost: { amount: "900", currency: "JPY" },
        total_includes_estimates: true,
        weather_warnings: ["Rain"],
      },
      labels: ["ai_recommended"],
      recommendation_reason: null,
    },
  ],
};

const LABELS = { "ev-taxi": "estimate" };

/** A copy of OPTIONS changed by `change`, which edits the copy in place. */
function optionsWith(change) {
  const copy = structuredClone(OPTIONS);
  change(copy, copy.options[0].itinerary);
  return copy;
}

describe("isRenderableResult", () => {
  test.each([
    ["clarification", { kind: "clarification", message: "m", questions: [{ question: "Where?" }] }],
    ["no feasible result", { kind: "no_feasible_result", message: "m", retry_question: "Later?" }],
    ["execution failure", { kind: "execution_failure", message: "m", reason: "provider_failure" }],
    ["validated options", OPTIONS],
  ])("accepts a complete %s", (_name, result) => {
    expect(isRenderableResult(result, LABELS, REQUEST)).toBe(true);
  });

  test.each([
    ["not an object", "options"],
    ["an unknown kind", { kind: "maybe", message: "m" }],
    ["questions not a list", { kind: "clarification", message: "m", questions: "Where?" }],
    ["a question without text", { kind: "clarification", message: "m", questions: [{}] }],
    ["no retry question", { kind: "no_feasible_result", message: "m" }],
  ])("rejects %s", (_name, result) => {
    expect(isRenderableResult(result, LABELS, REQUEST)).toBe(false);
  });

  test.each([
    ["options not a list", (r) => (r.options = "none")],
    ["no options", (r) => (r.options = [])],
    ["an unknown slot label", (r) => (r.options[0].labels = ["fastest"])],
    ["a numeric reason", (r) => (r.options[0].recommendation_reason = 3)],
    ["an option without stops", (_r, it) => delete it.candidate.stops],
    ["a stop without a label", (_r, it) => delete it.candidate.stops[0].label],
    ["one leg too few", (_r, it) => it.candidate.legs.pop()],
    ["an unknown leg mode", (_r, it) => (it.candidate.legs[0].mode = "boat")],
    ["a leg duration string", (_r, it) => (it.candidate.legs[0].duration_minutes = "10")],
    ["leg evidence not ids", (_r, it) => (it.candidate.legs[1].evidence_ids = "ev-taxi")],
    ["a cost amount of the wrong type", (_r, it) => (it.candidate.costs[0].amount = "JPY 900")],
    ["a cost without evidence ids", (_r, it) => delete it.candidate.costs[0].evidence_ids],
    ["a missing leave-by time", (_r, it) => it.leave_by.pop()],
    ["an unparseable time", (_r, it) => (it.arrival_at_commitment = "soon")],
    ["a numeric total amount", (_r, it) => (it.total_cost.amount = 900)],
    ["a string estimates flag", (_r, it) => (it.total_includes_estimates = "true")],
    ["warnings not text", (_r, it) => (it.weather_warnings = [1])],
  ])("rejects validated options with %s", (_name, change) => {
    expect(isRenderableResult(optionsWith(change), LABELS, REQUEST)).toBe(false);
  });

  test("rejects options whose estimate labels are not an object", () => {
    expect(isRenderableResult(OPTIONS, null, REQUEST)).toBe(false);
  });

  test.each([
    ["no request", undefined],
    ["no start location", { ...REQUEST, gap: { start_location: null } }],
    ["no commitment label", { ...REQUEST, commitment: { ...REQUEST.commitment, location: {} } }],
    [
      "an unknown commitment timezone",
      { ...REQUEST, commitment: { ...REQUEST.commitment, timezone: "Mars/Olympus" } },
    ],
  ])("rejects options rendered with %s", (_name, request) => {
    expect(isRenderableResult(OPTIONS, LABELS, request)).toBe(false);
  });
});

describe("isResultEvent", () => {
  const continuation = { schema_version: 1, messages: [], request: REQUEST };

  test("accepts a renderable result with a continuation and labels", () => {
    expect(
      isResultEvent({ type: "result", result: OPTIONS, continuation, estimate_labels: LABELS }),
    ).toBe(true);
  });

  test("rejects a missing continuation", () => {
    const result = { kind: "clarification", message: "m", questions: [{ question: "Where?" }] };
    expect(isResultEvent({ type: "result", result, estimate_labels: {} })).toBe(false);
  });

  test("rejects a result that cannot be rendered", () => {
    const result = { ...OPTIONS, options: "none" };
    expect(isResultEvent({ type: "result", result, continuation, estimate_labels: {} })).toBe(
      false,
    );
  });
});
