import { expect, test } from "./fixtures.js";

test("the page loads the app shell", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Travel agent" })).toBeVisible();
});

test("the page reaches the demo backend through the /api proxy", async ({ page }) => {
  await page.goto("/");
  const response = await page.evaluate(async () => {
    const reply = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": "audio/webm" },
      body: new Uint8Array([1, 2, 3]),
    });
    return { status: reply.status, body: await reply.text() };
  });
  expect(response.status).toBe(200);
  expect(JSON.parse(response.body)).toEqual({
    transcript:
      "I'm free from 3pm and have dinner in Omotesando at 7pm. " +
      "I have 3000 to spend and would like something quiet.",
  });
});
