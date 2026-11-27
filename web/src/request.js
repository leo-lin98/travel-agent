/** Pure construction of the /api/turn request body (interface-contract.md §2.1, §3). */

/** The continuation of a conversation with nothing stored yet. Never mutated. */
export const FRESH_CONTINUATION = {
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
};

/** The turn body: the continuation verbatim and the message exactly as typed or transcribed. */
export function turnBody(continuation, message, userTimezone, deviceLocation) {
  return {
    continuation,
    message,
    user_timezone: userTimezone,
    device_location: deviceLocation,
  };
}

/** The device location sent after the user shares it; user-reported, not verified (R6). */
export function deviceLocationFrom(coords) {
  return {
    label: "Current location",
    latitude: coords.latitude,
    longitude: coords.longitude,
    place_id: null,
  };
}
