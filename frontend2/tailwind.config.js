/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        border: '#122033',
        background: '#0b1220',
        foreground: '#E6FBFF',
        primary: {
          DEFAULT: '#00F3FF',
          dark: '#00B3CC',
          light: '#66FBFF',
        },
        accent: '#FF00D0',
        success: '#00FFA0',
        warning: '#FFB84D',
        error: '#FF3B6B',
        surface: {
          DEFAULT: '#0b1220',
          light: '#121628',
        },
        text: {
          primary: '#E6FBFF',
          secondary: '#9BD8FF',
        },
      },
    },
  },
  plugins: [],
};
