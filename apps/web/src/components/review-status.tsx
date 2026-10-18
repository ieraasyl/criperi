import type { SaveReview } from "@app/contracts";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tick02Icon, Cancel01Icon, CircleIcon } from "@hugeicons/core-free-icons";
import { Badge } from "@/components/ui/badge";
import { reviewLabels } from "@/lib/review";

export function ReviewIcon({ status }: { status?: SaveReview["status"] }) {
  return (
    <HugeiconsIcon
      icon={status === "confirmed" ? Tick02Icon : status === "rejected" ? Cancel01Icon : CircleIcon}
      strokeWidth={2}
      aria-hidden="true"
    />
  );
}

export function ReviewStatus({ status }: { status?: SaveReview["status"] }) {
  return (
    <Badge variant={status ?? "secondary"}>
      <ReviewIcon status={status} />
      {status ? reviewLabels[status] : "Не проверено"}
    </Badge>
  );
}
