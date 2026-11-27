import { renderToString } from "react-dom/server";
import { afterEach, expect, test, vi } from "vitest";

import App from "./App.jsx";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("App renders the heading and an empty conversation when nothing is stored", () => {
  vi.stubGlobal("sessionStorage", { getItem: () => null });
  const html = renderToString(<App />);
  expect(html).toContain("<h1>Travel agent</h1>");
  expect(html).toContain('role="log"');
  expect(html).not.toContain('data-testid="message"');
  expect(html).not.toContain('role="alert"');
});
