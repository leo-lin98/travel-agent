/** Builds the ordered itinerary list of a checked option from result fields only (R2). */
import { estimateLabelText, modeName } from "./format.js";

const legStep = (leg, estimateLabels) => ({
  kind: "leg",
  mode: modeName(leg.mode),
  minutes: leg.duration_minutes,
  estimate: estimateLabelText(leg.evidence_ids, estimateLabels),
});

/**
 * The start, then each leg followed by its stop, then the last leg and the commitment.
 * Times come from `leave_by` (start first) and `arrival_at_commitment`; nothing is computed.
 */
export function itinerarySteps(itinerary, startLabel, commitmentLabel, estimateLabels) {
  const { stops, legs } = itinerary.candidate;
  const start = { kind: "place", label: startLabel, verb: "Leave by", time: itinerary.leave_by[0] };
  const visits = stops.flatMap((stop, index) => [
    legStep(legs[index], estimateLabels),
    { kind: "place", label: stop.label, verb: "Leave by", time: itinerary.leave_by[index + 1] },
  ]);
  const arrival = {
    kind: "place",
    label: commitmentLabel,
    verb: "Arrive at",
    time: itinerary.arrival_at_commitment,
  };
  return [start, ...visits, legStep(legs[stops.length], estimateLabels), arrival];
}
