/**
 * Phase 3 browser acceptance: the browser's IANA timezone is sent on every turn, on both
 * sides of a daylight-saving change and in a zone without one, and itinerary times stay in the
 * commitment's timezone (R1; ui-contract.md §2, §5). No page.route is used.
 */
import { expect, test } from "./fixtures.js";
import {
  FIRST_MESSAGE,
  SECOND_MESSAGE,
  collectTurnBodies,
  completeTypedFlow,
  option,
  questions,
  sendTyped,
} from "./helpers.js";

// America/New_York leaves daylight saving time at 2026-11-01 06:00 UTC.
const BEFORE_CHANGE = new Date("2026-11-01T05:30:00Z");
const AFTER_CHANGE = new Date("2026-11-01T06:30:00Z");

const utcOffsetMinutes = (page) => page.evaluate(() => new Date().getTimezoneOffset());

/** Sends one turn at each instant; returns the browser's UTC offsets and the zones sent. */
async function turnsAcrossTheChange(page) {
  const bodies = collectTurnBodies(page);
  await page.clock.setFixedTime(BEFORE_CHANGE);
  await page.goto("/");
  const offsets = [await utcOffsetMinutes(page)];
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
  await page.clock.setFixedTime(AFTER_CHANGE);
  offsets.push(await utcOffsetMinutes(page));
  await sendTyped(page, SECOND_MESSAGE);
  return { offsets, zones: bodies.map((body) => body.user_timezone) };
}

test.describe("America/New_York", () => {
  test.use({ timezoneId: "America/New_York" });

  test("the IANA zone is sent on both sides of the daylight-saving change", async ({ page }) => {
    const { offsets, zones } = await turnsAcrossTheChange(page);

    expect(offsets).toEqual([240, 300]);
    expect(zones).toEqual(["America/New_York", "America/New_York"]);
  });

  test("itinerary times stay in the commitment's timezone", async ({ page }) => {
    await page.goto("/");
    await completeTypedFlow(page);

    const steps = option(page, "AI Recommended")
      .getByRole("list", { name: "Itinerary" })
      .getByRole("listitem");
    await expect(steps.first()).toContainText("16:25");
    await expect(steps.last()).toContainText("18:45");
  });
});

test.describe("Asia/Tokyo", () => {
  test.use({ timezoneId: "Asia/Tokyo" });

  test("the IANA zone is sent and the offset does not change", async ({ page }) => {
    const { offsets, zones } = await turnsAcrossTheChange(page);

    expect(offsets).toEqual([-540, -540]);
    expect(zones).toEqual(["Asia/Tokyo", "Asia/Tokyo"]);
  });
});
