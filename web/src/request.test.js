import { expect, test } from "vitest";

import { FRESH_CONTINUATION, deviceLocationFrom, turnBody } from "./request.js";

test("the fresh continuation is the contract's empty conversation", () => {
  expect(FRESH_CONTINUATION).toEqual({
    schema_version: 1,
    messages: [],
    request: {
      schema_version: 1,
      gap: { start_time: null, start_location: null },
      commitment: { location: null, time: null, timezone: null },
      budget: null,
      transport_modes: [],
      constraints: [],
      unresolved_questions: [],
    },
  });
});

test("the turn body has exactly the four contract keys, with the message unchanged", () => {
  const continuation = { anything: ["kept", "verbatim"] };
  const location = { label: "Current location", latitude: 1, longitude: 2, place_id: null };
  expect(turnBody(continuation, "  hi there ", "Asia/Tokyo", location)).toEqual({
    continuation,
    message: "  hi there ",
    user_timezone: "Asia/Tokyo",
    device_location: location,
  });
});

test("a device location carries the coordinates, a non-empty label, and no place id", () => {
  expect(deviceLocationFrom({ latitude: 35.6595, longitude: 139.7005, accuracy: 20 })).toEqual({
    label: "Current location",
    latitude: 35.6595,
    longitude: 139.7005,
    place_id: null,
  });
});
