/**
 * Foundational Tailwind CSS Configuration Block
 * Enforces the "Antigravity" Sports Design System for Pickleball Vision AI
 */
module.exports = {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./js/**/*.js"],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#090C10",
          light: "#F4F6F9"
        },
        surface: {
          DEFAULT: "#10151E",
          elevated: "#161D29",
          hover: "#1C2534"
        },
        border: {
          subtle: "#1D2636",
          strong: "#2C3950"
        },
        court: {
          optic: "#D4FF00",      // Pickleball Optic Lime
          blue: "#0078FF",       // Team A / Near Court
          teal: "#00D4B2",       // Team B / Far Court
          smash: "#FF5500",      // High-velocity Smash (> 22 px/f)
          playing: "#13334C",    // Regulation Playing Surface
          kitchen: "#0F263A",    // 7 ft Non-Volley Zone
          fault: "#FF3B30",      // Out / Kitchen Fault
          inbounds: "#28CD41"    // Official In Call
        }
      },
      fontFamily: {
        display: ["Urbanist", "Inter", "sans-serif"],
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"]
      },
      letterSpacing: {
        athletic: "-0.03em",
        caps: "0.08em"
      },
      boxShadow: {
        antigravity: "0 6px 24px -4px rgba(0, 0, 0, 0.5), 0 0 1px 0 rgba(255, 255, 255, 0.06) inset",
        crisp: "0 2px 10px -2px rgba(0, 0, 0, 0.35)"
      }
    }
  },
  plugins: []
};
