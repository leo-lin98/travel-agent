import { blockOutboundNetwork, expect, test } from "./fixtures.js";

test("the guard aborts and records a request to a host other than 127.0.0.1", async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ baseURL });
  const blocked = await blockOutboundNetwork(context);
  const page = await context.newPage();
  await page.goto("/");
  const failed = page.waitForEvent("requestfailed");
  await page.evaluate(() => fetch("https://outbound.invalid/").catch(() => null));
  expect((await failed).failure().errorText).toMatch(/^net::ERR_BLOCKED_BY_CLIENT/);
  expect(blocked).toEqual(["https://outbound.invalid/"]);
  await context.close();
});
