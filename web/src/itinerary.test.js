import { expect, test } from "vitest";

import { itinerarySteps } from "./itinerary.js";

const leg = (mode, minutes, evidenceIds) => ({
  mode,
  duration_minutes: minutes,
  distance_meters: 100,
  transfers: [],
  evidence_ids: evidenceIds,
});

const stop = (label) => ({
  place_id: null,
  label,
  latitude: 0,
  longitude: 0,
  fallback_key: null,
  visit_minutes: 30,
  evidence_ids: [],
});

const itinerary = {
  candidate: {
    candidate_id: "c",
    stops: [stop("Garden"), stop("Cafe")],
    legs: [
      leg("walking", 10, []),
      leg("taxi_rideshare", 7, ["ev-taxi"]),
      leg("public_transit", 5, []),
    ],
    costs: [],
  },
  leave_by: ["2026-11-20T15:00:00+09:00", "2026-11-20T16:00:00+09:00", "2026-11-20T17:00:00+09:00"],
  arrival_at_commitment: "2026-11-20T17:05:00+09:00",
  total_cost: { amount: "0", currency: "JPY" },
  total_includes_estimates: false,
  weather_warnings: [],
};

test("lists the start, each leg and stop in order, the last leg, and the commitment", () => {
  expect(itinerarySteps(itinerary, "Station", "Dinner", { "ev-taxi": "estimate" })).toEqual([
    { kind: "place", label: "Station", verb: "Leave by", time: "2026-11-20T15:00:00+09:00" },
    { kind: "leg", mode: "Walk", minutes: 10, estimate: null },
    { kind: "place", label: "Garden", verb: "Leave by", time: "2026-11-20T16:00:00+09:00" },
    { kind: "leg", mode: "Taxi or rideshare", minutes: 7, estimate: "Estimate" },
    { kind: "place", label: "Cafe", verb: "Leave by", time: "2026-11-20T17:00:00+09:00" },
    { kind: "leg", mode: "Public transit", minutes: 5, estimate: null },
    { kind: "place", label: "Dinner", verb: "Arrive at", time: "2026-11-20T17:05:00+09:00" },
  ]);
});
