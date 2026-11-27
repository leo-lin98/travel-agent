import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { MAX_CLIP_SECONDS, startClip } from "./microphone.js";

/** A fake MediaRecorder that emits one chunk and then "stop" when stopped. */
class FakeRecorder extends EventTarget {
  static created = [];

  constructor(stream) {
    super();
    this.stream = stream;
    this.state = "inactive";
    this.mimeType = "audio/webm;codecs=opus";
    FakeRecorder.created.push(this);
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    const data = new Event("dataavailable");
    data.data = new Blob([new Uint8Array([1, 2, 3])]);
    this.dispatchEvent(data);
    this.dispatchEvent(new Event("stop"));
  }
}

function fakeMicrophone() {
  const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
  const mediaDevices = { getUserMedia: vi.fn(async () => ({ getTracks: () => tracks })) };
  return { tracks, mediaDevices };
}

beforeEach(() => {
  FakeRecorder.created = [];
  vi.useFakeTimers();
  vi.stubGlobal("MediaRecorder", FakeRecorder);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

test("the clip limit is 30 seconds", () => {
  expect(MAX_CLIP_SECONDS).toBe(30);
});

test("asks for audio only and keeps the tracks open while recording", async () => {
  const { tracks, mediaDevices } = fakeMicrophone();
  await startClip(mediaDevices, MAX_CLIP_SECONDS);
  expect(mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
  expect(FakeRecorder.created[0].state).toBe("recording");
  tracks.forEach((track) => expect(track.stop).not.toHaveBeenCalled());
});

test("stopping ends the clip with the recorder's type and releases every track", async () => {
  const { tracks, mediaDevices } = fakeMicrophone();
  const session = await startClip(mediaDevices, MAX_CLIP_SECONDS);
  session.stop();
  const clip = await session.clip;
  expect(clip.type).toBe("audio/webm;codecs=opus");
  expect(clip.size).toBe(3);
  tracks.forEach((track) => expect(track.stop).toHaveBeenCalledTimes(1));
});

test("the recording stops by itself at the limit, not before", async () => {
  const { tracks, mediaDevices } = fakeMicrophone();
  const session = await startClip(mediaDevices, MAX_CLIP_SECONDS);
  vi.advanceTimersByTime(MAX_CLIP_SECONDS * 1000 - 1);
  expect(FakeRecorder.created[0].state).toBe("recording");
  vi.advanceTimersByTime(1);
  await session.clip;
  tracks.forEach((track) => expect(track.stop).toHaveBeenCalledTimes(1));
});

test("a stopped clip clears its limit timer", async () => {
  const { mediaDevices } = fakeMicrophone();
  const session = await startClip(mediaDevices, MAX_CLIP_SECONDS);
  session.stop();
  await session.clip;
  expect(vi.getTimerCount()).toBe(0);
});

test("a recorder that cannot be created releases the tracks and rejects", async () => {
  const { tracks, mediaDevices } = fakeMicrophone();
  const failure = new DOMException("unsupported", "NotSupportedError");
  vi.stubGlobal(
    "MediaRecorder",
    class {
      constructor() {
        throw failure;
      }
    },
  );
  await expect(startClip(mediaDevices, MAX_CLIP_SECONDS)).rejects.toBe(failure);
  tracks.forEach((track) => expect(track.stop).toHaveBeenCalledTimes(1));
});
