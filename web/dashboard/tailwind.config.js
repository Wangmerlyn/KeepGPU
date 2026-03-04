/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Manrope", "Segoe UI", "sans-serif"],
        serif: ["Newsreader", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"]
      }
    }
  },
  plugins: []
}
