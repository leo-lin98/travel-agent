/** Consent-based device location (R6, ui-contract.md §8). */
import { deviceLocationFrom } from "./request.js";

const DENIED = { state: "denied", deviceLocation: null };

/**
 * Asks `geolocation` for the position and calls `onResult` with the shared location or a
 * denial. Missing geolocation is treated as a denial, so the user is asked for the start.
 */
export function requestDeviceLocation(geolocation, onResult) {
  if (geolocation === undefined) {
    onResult(DENIED);
    return;
  }
  geolocation.getCurrentPosition(
    (position) =>
      onResult({ state: "shared", deviceLocation: deviceLocationFrom(position.coords) }),
    () => onResult(DENIED),
  );
}
