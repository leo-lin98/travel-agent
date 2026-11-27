import { expect, test } from "vitest";

import { appendTranscript } from "./draft.js";

test("an empty input becomes the transcript", () => {
  expect(appendTranscript("", "From Shibuya.")).toBe("From Shibuya.");
});

test("a transcript is appended after one space to existing text", () => {
  expect(appendTranscript("Draft answer", "From Shibuya.")).toBe("Draft answer From Shibuya.");
});
