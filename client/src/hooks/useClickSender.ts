import { useCallback } from "react";
import { usePipecatClient } from "@pipecat-ai/client-react";
import type { ClickEvent } from "../types";

type OutboundMessage = ClickEvent | { kind: "hello" };

export function useClickSender() {
  const client = usePipecatClient();
  return useCallback(
    (event: OutboundMessage) => {
      if (!client) return;
      // Send each click as a UI event named after its ``kind``; the rest
      // of the fields are the payload the UIWorker's @ui_event handlers
      // receive.
      const { kind, ...payload } = event;
      client.sendUIEvent(kind, payload);
    },
    [client],
  );
}
