/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
      "./templates/**/*.html",
      "./static/src/**/*.js",
      "./node_modules/flowbite/**/*.js"
  ],
  theme: {
    extend: {
      colors: {
				background: '#F5F5F5',
				background2: '#C2D9FF',
				primary: '#7752FE',
				seconday: '#8E8FFA',
				accent: '#190482',
				// text: '#190482',
			},
      fontFamily: {
        skranji: ['Skranji'],
        bree: ['Bree Serif', 'serif'],
        kdam: ['Kdam Thmor Pro', 'sans-serif'],
        proza: ['Proza Libre', 'sans-serif'],
        montserrat: ['Montserrat', 'sans-serif'],
      }
    },
  },
  plugins: [
    require("flowbite/plugin")
  ],
}
