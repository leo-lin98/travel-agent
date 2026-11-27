/** Pure editing of the message draft. */

/** Puts a transcript in the draft: the transcript alone, or appended after one space (UQ4). */
export function appendTranscript(draft, transcript) {
  return draft === "" ? transcript : `${draft} ${transcript}`;
}
