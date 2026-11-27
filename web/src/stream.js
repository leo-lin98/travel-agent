/** Pure parsing of the /api/turn NDJSON stream (interface-contract.md §2.1). */

/**
 * Parses each complete line of `buffer` as one JSON event and returns the unfinished remainder.
 * A line that is not JSON throws, so a broken stream is never silently skipped.
 */
export function takeEvents(buffer) {
  const lines = buffer.split("\n");
  const rest = lines.pop();
  return { events: lines.map((line) => JSON.parse(line)), rest };
}
