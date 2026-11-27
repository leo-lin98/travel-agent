import { describe, expect, test } from "vitest";

import {
  estimateLabelText,
  formatMoney,
  formatTime,
  modeName,
  slotName,
  totalCostText,
} from "./format.js";

describe("formatTime", () => {
  test("shows 24-hour HH:MM in the given zone, not the browser's", () => {
    expect(formatTime("2026-11-20T16:25:00+09:00", "Asia/Tokyo")).toBe("16:25");
    expect(formatTime("2026-11-20T16:25:00+09:00", "America/New_York")).toBe("02:25");
  });

  test("shows midnight as 00", () => {
    expect(formatTime("2026-11-20T00:05:00+09:00", "Asia/Tokyo")).toBe("00:05");
  });
});

describe("money", () => {
  test("is the currency code and the exact amount string", () => {
    expect(formatMoney({ amount: "1200.50", currency: "JPY" })).toBe("JPY 1200.50");
  });

  test("the total says when it includes estimates", () => {
    const total = { amount: "1200", currency: "JPY" };
    expect(totalCostText(total, false)).toBe("Estimated total cost: JPY 1200");
    expect(totalCostText(total, true)).toBe("Estimated total cost: JPY 1200 (includes estimates)");
  });
});

describe("estimateLabelText", () => {
  const labels = { "ev-a": "estimate", "ev-b": "ai_best_guess" };

  test("is null when no cited evidence is labelled", () => {
    expect(estimateLabelText(["ev-x"], labels)).toBeNull();
    expect(estimateLabelText([], labels)).toBeNull();
  });

  test("names the label of the cited evidence", () => {
    expect(estimateLabelText(["ev-x", "ev-a"], labels)).toBe("Estimate");
    expect(estimateLabelText(["ev-b"], labels)).toBe("AI best guess");
  });

  test("AI best guess wins when both apply", () => {
    expect(estimateLabelText(["ev-a", "ev-b"], labels)).toBe("AI best guess");
  });
});

test("slot names follow the option label", () => {
  expect(["ai_recommended", "cheapest", "best_travel"].map(slotName)).toEqual([
    "AI Recommended",
    "Cheapest",
    "Shortest Distance/Time",
  ]);
});

test("mode names follow the transport mode", () => {
  expect(["walking", "public_transit", "taxi_rideshare"].map(modeName)).toEqual([
    "Walk",
    "Public transit",
    "Taxi or rideshare",
  ]);
});
