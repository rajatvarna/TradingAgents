"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";

export default function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 2, staleTime: 600_000 },
        },
      })
  );
  return (
    <QueryClientProvider client={client}>
      <Tooltip.Provider>
        {children}
      </Tooltip.Provider>
    </QueryClientProvider>
  );
}
