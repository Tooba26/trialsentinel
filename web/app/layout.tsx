import type { Metadata } from "next";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "@fontsource/ibm-plex-sans-condensed/600.css";
import "./globals.css";
import { ThemeProvider } from "@/components/theme";

export const metadata: Metadata = {
  title: "TrialSentinel: will this clinical trial finish?",
  description:
    "Stress-test a Phase 2/3 clinical trial design against 75,000 registered trials. Get the chance of early termination and the reasons behind it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
