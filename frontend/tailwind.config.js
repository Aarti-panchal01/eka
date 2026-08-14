/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Named rather than inlined as hex, so a palette change is one edit and
        // `bg-card` reads as intent instead of as a colour.
        ink: "#0a0a0a", // app background
        sidebar: "#111111", // left rail
        card: "#1a1a1a", // surfaces
        edge: "#262626", // hairlines
        gold: {
          DEFAULT: "#f5a623",
          soft: "#f5a62319",
          mid: "#f5a62340",
        },
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.5), 0 8px 24px -12px rgba(0,0,0,0.7)",
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
      transitionDuration: { DEFAULT: "200ms" },
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
