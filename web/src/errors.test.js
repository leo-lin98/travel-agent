import { describe, expect, test } from "vitest";

import { errorFromBody, errorText, uiError } from "./errors.js";

describe("errorFromBody", () => {
  test("uses the backend's code and message", () => {
    const body = JSON.stringify({ code: "invalid_request", message: "That request is invalid." });
    expect(errorFromBody(body)).toEqual({
      code: "invalid_request",
      message: "That request is invalid.",
    });
  });

  test.each(["<html>Bad gateway</html>", "{}", '{"code": 5, "message": "x"}', "null"])(
    "a body without a code and message, %s, is a network error",
    (body) => {
      expect(errorFromBody(body)).toEqual(uiError("network_error"));
    },
  );
});

describe("errorText", () => {
  test("is the backend's message", () => {
    expect(errorText({ code: "internal_error", message: "Something went wrong." })).toBe(
      "Something went wrong.",
    );
  });

  test("asks for a new conversation on overflow", () => {
    expect(errorText({ code: "continuation_overflow", message: "Too long." })).toContain(
      "Start a new conversation",
    );
  });

  test.each(["microphone_denied", "network_error", "stored_context_unreadable", "internal_error"])(
    "the UI code %s has its own text",
    (code) => {
      expect(uiError(code).code).toBe(code);
      expect(errorText(uiError(code))).toMatch(/\S/);
    },
  );

  test("the microphone error mentions the microphone", () => {
    expect(errorText(uiError("microphone_denied"))).toMatch(/microphone/i);
  });
});
