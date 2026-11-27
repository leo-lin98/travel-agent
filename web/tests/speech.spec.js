/**
 * Phase 3 browser acceptance: microphone denial, transcription failure, and the clip duration
 * limit (ui-contract.md §4, §6). The transcription-failure test injects the backend's error
 * with page.route, because the demo transcriber cannot fail for a non-empty clip.
 */
import { expect, test } from "./fixtures.js";
import {
  DEMO,
  FIRST_MESSAGE,
  MAX_CLIP_MS,
  alert,
  denyMicrophone,
  input,
  messages,
  questions,
  sendTyped,
  useSyntheticMicrophone,
} from "./helpers.js";

test("a denied microphone is explained and typing still works", async ({ page }) => {
  await denyMicrophone(page);
  await page.goto("/");

  await page.getByRole("button", { name: "Record message" }).click();

  await expect(alert(page)).toHaveAttribute("data-error-code", "microphone_denied");
  await expect(alert(page)).toContainText(/microphone/i);
  await expect(page.getByRole("button", { name: "Record message" })).toBeVisible();
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
});

test.describe("with a synthetic microphone", () => {
  test.beforeEach(async ({ page }) => {
    await useSyntheticMicrophone(page);
  });

  test("a transcription failure is shown and keeps the conversation and the draft", async ({
    page,
  }) => {
    await page.goto("/");
    await sendTyped(page, FIRST_MESSAGE);
    await expect(messages(page)).toHaveCount(2);
    await page.route("**/api/transcribe", (route) =>
      route.fulfill({
        status: 502,
        contentType: "application/json",
        body: JSON.stringify({
          code: "transcription_failed",
          message: "We could not transcribe that recording. Please try again or type it.",
        }),
      }),
    );
    await input(page).fill("Draft answer");

    await page.getByRole("button", { name: "Record message" }).click();
    await page.getByRole("button", { name: "Stop recording" }).click();

    await expect(alert(page)).toHaveAttribute("data-error-code", "transcription_failed");
    await expect(input(page)).toHaveValue("Draft answer");
    await expect(messages(page)).toHaveText([FIRST_MESSAGE, DEMO.clarification]);
  });

  test("a recording stops by itself at the clip duration limit and is transcribed", async ({
    page,
  }) => {
    await page.clock.install();
    await page.goto("/");

    await page.getByRole("button", { name: "Record message" }).click();
    await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
    await page.clock.fastForward(MAX_CLIP_MS - 1000);
    await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
    await page.clock.fastForward(1000);

    await expect(page.getByRole("button", { name: "Record message" })).toBeVisible();
    await expect(input(page)).toHaveValue(DEMO.transcript);
  });
});
