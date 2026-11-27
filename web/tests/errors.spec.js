/**
 * Phase 3 browser acceptance: malformed or tampered stored context, continuation overflow,
 * a mid-stream error, results the demo cannot produce, and the message length limit
 * (ui-contract.md §1, §2, §6, §7). Two groups use page.route to fulfil /api/turn: the results
 * the demo cannot produce (no-feasible, execution failure) and malformed result events (TR2).
 */
import { expect, test } from "./fixtures.js";
import {
  DEMO,
  FIRST_MESSAGE,
  FRESH_CONTINUATION,
  MAX_MESSAGE_CHARS,
  STORAGE_KEY,
  alert,
  completeTypedFlow,
  input,
  messages,
  options,
  questions,
  readStorage,
  seedStorage,
  sendTyped,
} from "./helpers.js";

const NEXT_MESSAGE = "Plan my afternoon.";

function storedWith(messageList, extra) {
  return JSON.stringify({
    continuation: { ...FRESH_CONTINUATION, messages: messageList, ...extra },
    last_result: null,
  });
}

function chat(count) {
  return Array.from({ length: count }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    text: `Message ${i}.`,
  }));
}

test("tampered stored context shows the backend's error and keeps the stored state", async ({
  page,
}) => {
  const tampered = storedWith(
    [
      { role: "user", text: "Earlier question" },
      { role: "assistant", text: "Earlier answer" },
    ],
    { evidence: [{ finding: "verified by the browser" }] },
  );
  await seedStorage(page, tampered);
  await page.goto("/");
  await expect(messages(page)).toHaveText(["Earlier question", "Earlier answer"]);

  await sendTyped(page, NEXT_MESSAGE);

  await expect(alert(page)).toHaveAttribute("data-error-code", "invalid_request");
  await expect(messages(page)).toHaveText(["Earlier question", "Earlier answer"]);
  await expect(input(page)).toHaveValue(NEXT_MESSAGE);
  expect(await readStorage(page)).toBe(tampered);
});

test("unreadable stored context is reported, not silently replaced", async ({ page }) => {
  await seedStorage(page, "{not json");
  await page.goto("/");

  await expect(alert(page)).toHaveAttribute("data-error-code", "stored_context_unreadable");
  await input(page).fill(NEXT_MESSAGE);
  await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();
  expect(await readStorage(page)).toBe("{not json");

  await page.getByRole("button", { name: "New conversation" }).click();
  await expect(alert(page)).toHaveCount(0);
  expect(await readStorage(page)).toBeNull();
  await sendTyped(page, FIRST_MESSAGE);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
});

test("a full conversation asks the user to start a new one and keeps it visible", async ({
  page,
}) => {
  const full = storedWith(chat(39), {});
  await seedStorage(page, full);
  await page.goto("/");
  await expect(messages(page)).toHaveCount(39);

  await sendTyped(page, NEXT_MESSAGE);

  await expect(alert(page)).toHaveAttribute("data-error-code", "continuation_overflow");
  await expect(alert(page)).toContainText("Start a new conversation");
  await expect(messages(page)).toHaveCount(39);
  expect(await readStorage(page)).toBe(full);

  await page.getByRole("button", { name: "New conversation" }).click();
  await expect(messages(page)).toHaveCount(0);
  expect(await readStorage(page)).toBeNull();
});

test("an error after streaming starts keeps the conversation, the options, and the draft", async ({
  page,
}) => {
  await page.goto("/");
  await completeTypedFlow(page);
  const stored = await readStorage(page);

  // The demo script has two turns; a third ends its stream with an internal_error event.
  await sendTyped(page, "And after dinner?");

  await expect(alert(page)).toHaveAttribute("data-error-code", "internal_error");
  await expect(messages(page)).toHaveCount(4);
  await expect(options(page).getByRole("article")).toHaveCount(3);
  await expect(input(page)).toHaveValue("And after dinner?");
  expect(await readStorage(page)).toBe(stored);
});

const INJECTED = {
  no_feasible_result: {
    kind: "no_feasible_result",
    message: "No plan fits your time and budget.",
    retry_question: "Would you like me to try again with a later dinner?",
  },
  execution_failure: {
    kind: "execution_failure",
    message: "A service I rely on failed before I could finish. Please try again.",
    reason: "provider_failure",
  },
};

for (const [kind, result] of Object.entries(INJECTED)) {
  test(`a ${kind} result is shown after the earlier conversation`, async ({ page }) => {
    await page.goto("/");
    await sendTyped(page, FIRST_MESSAGE);
    await expect(messages(page)).toHaveCount(2);
    await page.route("**/api/turn", (route) => {
      const body = route.request().postDataJSON();
      const continuation = {
        ...body.continuation,
        messages: [
          ...body.continuation.messages,
          { role: "user", text: body.message },
          { role: "assistant", text: result.message },
        ],
      };
      const lines = [
        { type: "status", status: "Researching" },
        { type: "result", result, continuation, estimate_labels: {} },
      ];
      return route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: lines.map((line) => `${JSON.stringify(line)}\n`).join(""),
      });
    });

    await sendTyped(page, NEXT_MESSAGE);

    await expect(messages(page)).toHaveText([
      FIRST_MESSAGE,
      DEMO.clarification,
      NEXT_MESSAGE,
      result.message,
    ]);
    const retry = page.getByTestId("retry-question");
    if (kind === "no_feasible_result") {
      await expect(retry).toHaveText(result.retry_question);
    } else {
      await expect(retry).toHaveCount(0);
    }
    await expect(alert(page)).toHaveCount(0);
    await expect(options(page)).toHaveCount(0);
    expect(JSON.parse(await readStorage(page)).last_result.result).toEqual(result);
  });
}

test("a message is bounded at the limit and a message at the limit is accepted", async ({
  page,
}) => {
  await page.goto("/");

  await input(page).fill("a".repeat(MAX_MESSAGE_CHARS + 1));
  await expect(input(page)).toHaveValue("a".repeat(MAX_MESSAGE_CHARS));
  const body = await sendTyped(page, "a".repeat(MAX_MESSAGE_CHARS));

  expect(body.message).toHaveLength(MAX_MESSAGE_CHARS);
  await expect(questions(page).getByRole("listitem")).toHaveCount(2);
  await expect(alert(page)).toHaveCount(0);
});

const BROKEN_STORED_RESULTS = ["option_without_stops", "cost_amount_of_the_wrong_type"];

for (const kind of BROKEN_STORED_RESULTS) {
  test(`a stored result with ${kind.replaceAll("_", " ")} is unreadable, not a blank page`, async ({
    page,
  }) => {
    await page.goto("/");
    await completeTypedFlow(page);
    await page.evaluate(
      ([key, broken]) => {
        const value = JSON.parse(globalThis.sessionStorage.getItem(key));
        const candidate = value.last_result.result.options[0].itinerary.candidate;
        if (broken === "option_without_stops") {
          delete candidate.stops;
        } else {
          candidate.costs[0].amount = "JPY 500";
        }
        globalThis.sessionStorage.setItem(key, JSON.stringify(value));
      },
      [STORAGE_KEY, kind],
    );
    const broken = await readStorage(page);

    await page.reload();

    await expect(page.getByRole("heading", { name: "Travel agent" })).toBeVisible();
    await expect(alert(page)).toHaveAttribute("data-error-code", "stored_context_unreadable");
    await input(page).fill(NEXT_MESSAGE);
    await expect(page.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(await readStorage(page)).toBe(broken);

    await page.getByRole("button", { name: "New conversation" }).click();
    await expect(alert(page)).toHaveCount(0);
    expect(await readStorage(page)).toBeNull();
  });
}

const MALFORMED_RESULT_EVENTS = {
  options_not_a_list: (continuation) => ({
    type: "result",
    result: { kind: "validated_options", message: "Here are your options.", options: "none" },
    continuation,
    estimate_labels: {},
  }),
  missing_continuation: () => ({
    type: "result",
    result: {
      kind: "clarification",
      message: "Where will you start?",
      questions: [{ topic: "start_location", question: "Where will you start?" }],
    },
    estimate_labels: {},
  }),
};

for (const [kind, finalEvent] of Object.entries(MALFORMED_RESULT_EVENTS)) {
  test(`a result event with ${kind.replaceAll("_", " ")} is an internal error that keeps state`, async ({
    page,
  }) => {
    await page.goto("/");
    await sendTyped(page, FIRST_MESSAGE);
    await expect(messages(page)).toHaveCount(2);
    const stored = await readStorage(page);
    await page.route("**/api/turn", (route) => {
      const body = route.request().postDataJSON();
      const continuation = {
        ...body.continuation,
        messages: [
          ...body.continuation.messages,
          { role: "user", text: body.message },
          { role: "assistant", text: "Here are your options." },
        ],
      };
      const lines = [{ type: "status", status: "Researching" }, finalEvent(continuation)];
      return route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        body: lines.map((line) => `${JSON.stringify(line)}\n`).join(""),
      });
    });

    await sendTyped(page, NEXT_MESSAGE);

    await expect(page.getByRole("heading", { name: "Travel agent" })).toBeVisible();
    await expect(alert(page)).toHaveAttribute("data-error-code", "internal_error");
    await expect(messages(page)).toHaveText([FIRST_MESSAGE, DEMO.clarification]);
    await expect(input(page)).toHaveValue(NEXT_MESSAGE);
    await expect(page.getByRole("button", { name: "Send" })).toBeEnabled();
    await expect(page.getByRole("button", { name: "Record message" })).toBeEnabled();
    expect(await readStorage(page)).toBe(stored);
  });
}
