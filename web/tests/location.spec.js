/**
 * Phase 3 browser acceptance: consent-based device location (R9, O1; ui-contract.md §8).
 * Geolocation is mocked through the Playwright context; no page.route is used.
 */
import { expect, test } from "./fixtures.js";
import {
  FIRST_MESSAGE,
  SECOND_MESSAGE,
  collectTurnBodies,
  options,
  questions,
  recordClip,
  send,
  sendTyped,
  useSyntheticMicrophone,
} from "./helpers.js";

const shareButton = (page) => page.getByRole("button", { name: "Share my location" });

test.describe("location permission granted", () => {
  test.use({
    permissions: ["geolocation"],
    geolocation: { latitude: 35.6595, longitude: 139.7005 },
  });

  test("the device location is sent only after the user shares it", async ({ page }) => {
    const bodies = collectTurnBodies(page);
    await page.goto("/");

    await sendTyped(page, FIRST_MESSAGE);
    await shareButton(page).click();
    await expect(shareButton(page)).toHaveAttribute("aria-pressed", "true");
    await sendTyped(page, SECOND_MESSAGE);

    expect(bodies[0].device_location).toBeNull();
    expect(bodies[1].device_location).toEqual({
      label: expect.any(String),
      latitude: 35.6595,
      longitude: 139.7005,
      place_id: null,
    });
    expect(bodies[1].device_location.label.trim()).not.toBe("");
  });
});

test.describe("location permission denied", () => {
  test.beforeEach(async ({ page }) => {
    await useSyntheticMicrophone(page);
  });

  test("a denied location asks for the start, which can be answered by voice", async ({ page }) => {
    const bodies = collectTurnBodies(page);
    await page.goto("/");

    await shareButton(page).click();
    await expect(shareButton(page)).toHaveAttribute("aria-pressed", "false");
    await expect(page.getByTestId("location-notice")).toContainText("starting location");

    await sendTyped(page, FIRST_MESSAGE);
    await expect(questions(page).getByRole("listitem").first()).toHaveText("Where will you start?");
    await recordClip(page);
    await send(page);

    await expect(options(page).getByRole("article")).toHaveCount(3);
    expect(bodies.map((body) => body.device_location)).toEqual([null, null]);
  });
});
