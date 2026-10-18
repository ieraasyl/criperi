import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function FindingCode({ code, title }: { code?: string; title: string }) {
  if (!code) return null;
  return (
    <Tooltip>
      <TooltipTrigger
        render={<Badge variant="secondary" tabIndex={0} className="cursor-help align-middle" />}
      >
        {code}
      </TooltipTrigger>
      <TooltipContent>
        {code}: {title}. Код категории события.
      </TooltipContent>
    </Tooltip>
  );
}
