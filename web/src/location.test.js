import { expect, test } from "vitest";

import { requestDeviceLocation } from "./location.js";

const DENIED = { state: "denied", deviceLocation: null };

test("a granted position is shared as the device location", () => {
  const results = [];
  const geolocation = {
    getCurrentPosition: (onSuccess) => onSuccess({ coords: { latitude: 35.6, longitude: 139.7 } }),
  };
  requestDeviceLocation(geolocation, (result) => results.push(result));
  expect(results).toEqual([
    {
      state: "shared",
      deviceLocation: {
        label: "Current location",
        latitude: 35.6,
        longitude: 139.7,
        place_id: null,
      },
    },
  ]);
});

test("a denied or failed request is a denial", () => {
  const results = [];
  const geolocation = {
    getCurrentPosition: (_onSuccess, onError) => onError({ code: 1 }),
  };
  requestDeviceLocation(geolocation, (result) => results.push(result));
  expect(results).toEqual([DENIED]);
});

test("missing geolocation is a denial, not a thrown error", () => {
  const results = [];
  requestDeviceLocation(undefined, (result) => results.push(result));
  expect(results).toEqual([DENIED]);
});
