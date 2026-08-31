export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['"Geist Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        ink: { DEFAULT: '#0E1420', 2: '#535C6C', 3: '#8A93A3' },
        line: { DEFAULT: '#DFE3EA', 2: '#C9CFDA' },
        panel: { DEFAULT: '#FFFFFF', 2: '#F7F8FA', 3: '#EDEFF3' },
        acc: { DEFAULT: '#4338CA', soft: '#EEEDFB', line: '#C7C4F0' },
        dom: { finance: '#047857', hr: '#A21CAF', legal: '#B45309', engineering: '#0369A1' },
        ok: { DEFAULT: '#047857', bg: '#E6F4EF' },
        warn: { DEFAULT: '#B45309', bg: '#FBF0E2' },
        bad: { DEFAULT: '#B91C1C', bg: '#FBE9E9' },
      },
    },
  },
  plugins: [],
}
