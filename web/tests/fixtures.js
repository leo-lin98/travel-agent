/**
 * Shared Playwright fixtures; every browser test imports `test` and `expect` from here.
 *
 * The outbound-network guard routes browser HTTP(S) requests in each test's `context`. It does
 * not cover, so tests must not use:
 * - a page-level `page.route` handler that calls `route.continue()` (use `route.fallback()`);
 * - the `request` fixture, `page.request`, or Node-side `fetch` in test code;
 * - WebSockets or service-worker requests.
 */
import { expect, test as base } from "@playwright/test";

import { HOST } from "../offline-servers.js";

/** Aborts every request in `context` to a host other than 127.0.0.1; returns their URLs. */
export async function blockOutboundNetwork(context) {
  const blocked = [];
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (new URL(url).hostname === HOST) {
      return route.fallback();
    }
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
  return blocked;
}

export const test = base.extend({
  /** Applies the guard to the test's context and fails the test if any request was blocked. */
  outboundNetworkGuard: [
    async ({ context }, use) => {
      const blocked = await blockOutboundNetwork(context);
      await use();
      expect(blocked, `requests to hosts other than ${HOST}`).toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
