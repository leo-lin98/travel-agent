/**
 * One button-controlled recording (ui-contract.md §4, D3). The microphone is opened on the
 * press, records a single clip, and its tracks are stopped as soon as the clip ends.
 */

/** Provisional clip duration limit in seconds (O2); the backend enforces bytes. */
export const MAX_CLIP_SECONDS = 30;

/**
 * Opens the microphone and starts recording one clip. Resolves with `stop`, which ends the clip
 * early, and `clip`, a promise of the recorded Blob. The clip ends by itself after `maxSeconds`.
 * Rejects when the microphone is denied or unavailable, with no track left open.
 */
export async function startClip(mediaDevices, maxSeconds) {
  const stream = await mediaDevices.getUserMedia({ audio: true });
  const releaseTracks = () => stream.getTracks().forEach((track) => track.stop());
  try {
    const recorder = new MediaRecorder(stream);
    const stop = () => {
      if (recorder.state !== "inactive") {
        recorder.stop();
      }
    };
    const timer = setTimeout(stop, maxSeconds * 1000);
    const clip = new Promise((resolve) => {
      const chunks = [];
      recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
      recorder.addEventListener("stop", () => {
        clearTimeout(timer);
        releaseTracks();
        resolve(new Blob(chunks, { type: recorder.mimeType }));
      });
    });
    recorder.start();
    return { stop, clip };
  } catch (error) {
    releaseTracks();
    throw error;
  }
}
