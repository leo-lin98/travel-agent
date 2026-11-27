/**
 * Phase 3 browser acceptance: the planning flow by typing, by voice, and mixed; live status;
 * multiple turns, refresh, fresh session, and session restoration (ui-contract.md §1–§5, §7).
 */
import { expect, test } from "./fixtures.js";
import {
  DEMO,
  FIRST_MESSAGE,
  FRESH_CONTINUATION,
  SECOND_MESSAGE,
  collectTurnBodies,
  completeTypedFlow,
  conversation,
  lastEvent,
  logTurnStatus,
  messages,
  option,
  options,
  paceTurnStream,
  questions,
  readStorage,
  recordClip,
  send,
  seedStorage,
  sendTyped,
  shownStatusesPerTurn,
  useSyntheticMicrophone,
  userMessages,
} from "./helpers.js";

const SLOTS = ["AI Recommended", "Cheapest", "Shortest Distance/Time"];

test("typed messages complete the planning flow with distinct options and every R2 field", async ({
  page,
}) => {
  const turnResponses = [];
  page.on("response", (response) => {
    if (response.url().endsWith("/api/turn")) {
      turnResponses.push(response);
    }
  });
  await page.goto("/");

  const first = await sendTyped(page, FIRST_MESSAGE);
  await expect(messages(page)).toHaveText([FIRST_MESSAGE, DEMO.clarification]);
  await expect(questions(page).getByRole("listitem")).toHaveText(DEMO.questions);
  await expect(options(page)).toHaveCount(0);

  const second = await sendTyped(page, SECOND_MESSAGE);
  const firstEvent = lastEvent(await turnResponses[0].text());
  expect(first.continuation).toEqual(FRESH_CONTINUATION);
  expect(second.continuation).toEqual(firstEvent.continuation);

  await expect(messages(page)).toHaveText([
    FIRST_MESSAGE,
    DEMO.clarification,
    SECOND_MESSAGE,
    DEMO.optionsReply,
  ]);
  await expect(questions(page)).toHaveCount(0);
  await expect(options(page).getByRole("article")).toHaveCount(3);
  for (const slot of SLOTS) {
    await expect(option(page, slot)).toBeVisible();
  }

  const garden = option(page, "AI Recommended");
  const gardenSteps = garden.getByRole("list", { name: "Itinerary" }).getByRole("listitem");
  await expect(gardenSteps).toHaveText([
    /Shibuya Station.*Leave by.*16:25/,
    /Walk.*25 min/,
    /Demo Quiet Garden.*Leave by.*18:20/,
    /Walk.*25 min/,
    /Dinner in Omotesando.*Arrive at.*18:45/,
  ]);
  await expect(gardenSteps.nth(0).locator("time")).toHaveAttribute(
    "datetime",
    "2026-11-20T16:25:00+09:00",
  );
  await expect(gardenSteps.nth(4).locator("time")).toHaveAttribute(
    "datetime",
    "2026-11-20T18:45:00+09:00",
  );
  await expect(garden).toContainText("The quietest option, with time to linger before dinner.");
  await expect(garden).not.toContainText("Among the remaining plans");
  await expect(garden.getByTestId("total-cost")).toHaveText("Estimated total cost: JPY 500");
  await expect(garden.getByTestId("weather-warning")).toHaveText([
    "Rain is forecast during the outing; plan for wet conditions.",
  ]);
  await expect(garden).not.toContainText("AI best guess");

  const tea = option(page, "Cheapest");
  await expect(tea.getByRole("list", { name: "Itinerary" }).getByRole("listitem")).toHaveText([
    /Shibuya Station.*Leave by.*17:15/,
    /Walk.*10 min/,
    /Demo Tea House.*Leave by.*18:25/,
    /Walk.*20 min/,
    /Dinner in Omotesando.*Arrive at.*18:45/,
  ]);
  await expect(tea).toContainText("Among the remaining plans");
  await expect(tea.getByTestId("total-cost")).toHaveText("Estimated total cost: JPY 800");
  await expect(tea.getByTestId("weather-warning")).toHaveCount(0);

  const museum = option(page, "Shortest Distance/Time");
  await expect(museum.getByRole("list", { name: "Itinerary" }).getByRole("listitem")).toHaveText([
    /Shibuya Station.*Leave by.*16:15/,
    /Walk.*15 min/,
    /Demo Print Museum.*Leave by.*18:00/,
    /Walk.*12 min/,
    /Dinner in Omotesando.*Arrive at.*18:12/,
  ]);
  await expect(museum).toContainText("Among the remaining plans");
  await expect(museum.getByTestId("total-cost")).toHaveText(
    "Estimated total cost: JPY 1200 (includes estimates)",
  );
  await expect(museum.getByRole("list", { name: "Costs" }).getByRole("listitem")).toHaveText([
    /Demo Print Museum entry: JPY 1200.*AI best guess/,
  ]);

  const stored = JSON.parse(await readStorage(page));
  const secondEvent = lastEvent(await turnResponses[1].text());
  expect(Object.keys(stored).sort()).toEqual(["continuation", "last_result"]);
  expect(stored.continuation).toEqual(secondEvent.continuation);
  expect(stored.last_result).toEqual({
    result: secondEvent.result,
    estimate_labels: secondEvent.estimate_labels,
  });
});

test.describe("spoken messages", () => {
  test.beforeEach(async ({ page }) => {
    await useSyntheticMicrophone(page);
  });

  test("spoken messages alone complete the planning flow; transcripts are never auto-sent", async ({
    page,
  }) => {
    const bodies = collectTurnBodies(page);
    await page.goto("/");

    await expect(await recordClip(page)).toHaveValue(DEMO.transcript);
    await page.waitForTimeout(500);
    expect(bodies).toHaveLength(0);
    await send(page);
    await expect(questions(page).getByRole("listitem")).toHaveCount(2);

    await expect(await recordClip(page)).toHaveValue(DEMO.transcript);
    await send(page);

    await expect(options(page).getByRole("article")).toHaveCount(3);
    expect(bodies.map((body) => body.message)).toEqual([DEMO.transcript, DEMO.transcript]);
    await expect(userMessages(page)).toHaveText([DEMO.transcript, DEMO.transcript]);
  });

  test("a conversation can switch between typed and spoken messages", async ({ page }) => {
    await page.goto("/");

    await sendTyped(page, FIRST_MESSAGE);
    await expect(questions(page).getByRole("listitem")).toHaveCount(2);
    await recordClip(page);
    await send(page);

    await expect(options(page).getByRole("article")).toHaveCount(3);
    await expect(userMessages(page)).toHaveText([FIRST_MESSAGE, DEMO.transcript]);
  });
});

test("the status follows the agent live, back and forth, and a clarification never shows Generating itinerary", async ({
  page,
}) => {
  await paceTurnStream(page, 300);
  await logTurnStatus(page);
  await page.goto("/");

  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
  await expect.poll(() => shownStatusesPerTurn(page)).toEqual([["Researching"]]);

  await sendTyped(page, SECOND_MESSAGE);
  await expect(options(page).getByRole("article")).toHaveCount(3);
  await expect
    .poll(() => shownStatusesPerTurn(page))
    .toEqual([
      ["Researching"],
      ["Researching", "Planning", "Researching", "Planning", "Generating itinerary"],
    ]);
});

test("a refresh keeps the conversation, and the next turn continues it", async ({ page }) => {
  await page.goto("/");
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);

  await page.reload();
  await expect(messages(page)).toHaveText([FIRST_MESSAGE, DEMO.clarification]);
  await expect(questions(page).getByRole("listitem")).toHaveText(DEMO.questions);

  await sendTyped(page, SECOND_MESSAGE);
  await expect(options(page).getByRole("article")).toHaveCount(3);

  await page.reload();
  await expect(messages(page)).toHaveCount(4);
  await expect(options(page).getByRole("article")).toHaveCount(3);
  await expect(option(page, "AI Recommended").getByTestId("total-cost")).toHaveText(
    "Estimated total cost: JPY 500",
  );
});

test("a new tab starts a fresh conversation, and restored session storage restores it", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await completeTypedFlow(page);
  const stored = await readStorage(page);

  const fresh = await context.newPage();
  const freshBodies = collectTurnBodies(fresh);
  await fresh.goto("/");
  await expect(conversation(fresh)).toBeVisible();
  await expect(messages(fresh)).toHaveCount(0);
  await expect(options(fresh)).toHaveCount(0);
  expect(await readStorage(fresh)).toBeNull();
  await sendTyped(fresh, FIRST_MESSAGE);
  expect(freshBodies[0].continuation).toEqual(FRESH_CONTINUATION);

  const restored = await context.newPage();
  await seedStorage(restored, stored);
  await restored.goto("/");
  await expect(messages(restored)).toHaveText(await messages(page).allTextContents());
  await expect(options(restored).getByRole("article")).toHaveCount(3);
  expect(await readStorage(restored)).toBe(stored);
});
