/** Pure display formatting for checked itinerary values (ui-contract.md §5, UQ3). */

/** Formats an ISO datetime as 24-hour HH:MM in the IANA zone `timeZone`. */
export function formatTime(iso, timeZone) {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone,
  }).format(new Date(iso));
}

/** Formats a Money value as "<currency> <amount>", keeping the amount string as received. */
export function formatMoney(money) {
  return `${money.currency} ${money.amount}`;
}

/** The total-cost line, flagged when the total includes estimates (R13). */
export function totalCostText(total, includesEstimates) {
  const text = `Estimated total cost: ${formatMoney(total)}`;
  return includesEstimates ? `${text} (includes estimates)` : text;
}

const SLOT_NAMES = {
  ai_recommended: "AI Recommended",
  cheapest: "Cheapest",
  best_travel: "Shortest Distance/Time",
};

const MODE_NAMES = {
  walking: "Walk",
  public_transit: "Public transit",
  taxi_rideshare: "Taxi or rideshare",
};

/** The display name of an option slot label. */
export function slotName(label) {
  return SLOT_NAMES[label];
}

/** The display name of a transport mode. */
export function modeName(mode) {
  return MODE_NAMES[mode];
}

/**
 * The estimate label for an item citing `evidenceIds`, or null when none is labelled.
 * "AI best guess" wins over "Estimate" (ui-contract.md §5).
 */
export function estimateLabelText(evidenceIds, estimateLabels) {
  const labels = evidenceIds
    .filter((id) => Object.hasOwn(estimateLabels, id))
    .map((id) => estimateLabels[id]);
  if (labels.includes("ai_best_guess")) {
    return "AI best guess";
  }
  return labels.includes("estimate") ? "Estimate" : null;
}
