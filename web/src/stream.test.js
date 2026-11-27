import { expect, test } from "vitest";

import { takeEvents } from "./stream.js";

const status = { type: "status", status: "Planning" };
const line = (event) => `${JSON.stringify(event)}\n`;

test("parses every complete line and keeps an unfinished line for the next chunk", () => {
  const partial = line({ type: "result" }).slice(0, 5);
  expect(takeEvents(line(status) + line(status) + partial)).toEqual({
    events: [status, status],
    rest: partial,
  });
});

test("a line completed by a later chunk is parsed then", () => {
  const whole = line(status);
  const first = takeEvents(whole.slice(0, 10));
  expect(first).toEqual({ events: [], rest: whole.slice(0, 10) });
  expect(takeEvents(first.rest + whole.slice(10))).toEqual({ events: [status], rest: "" });
});

test("an unparseable line throws instead of being skipped", () => {
  expect(() => takeEvents("{not json\n")).toThrow(SyntaxError);
});
