/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Design tokens. Named rather than hex-inline so a palette change is
        // one edit, and so `bg-ink` reads as intent instead of as a colour.
        ink: "#0a0a0a", // app background
        card: "#1a1a1a", // surfaces
        edge: "#262626", // hairlines
        gold: {
          DEFAULT: "#f5a623",
          soft: "#f5a62319", // 10% — tints, active rows
          mid: "#f5a62340", // 25% — borders on active state
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseRing: {
          "0%": { boxShadow: "0 0 0 0 rgba(239,68,68,0.45)" },
          "100%": { boxShadow: "0 0 0 12px rgba(239,68,68,0)" },
        },
      },
      animation: {
        rise: "rise 160ms ease-out",
        pulseRing: "pulseRing 1.4s ease-out infinite",
      },
    },
  },
  plugins: [],
};
