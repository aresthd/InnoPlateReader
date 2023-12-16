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
				secondary: '#8E8FFA',
				accent: '#190482',
				text: '#F5F5F5',
				text2: '#190482',
				// text: '#190482',
			},
      fontFamily: {
        skranji: ['Skranji'],
        bree: ['Bree Serif', 'serif'],
        kdam: ['Kdam Thmor Pro', 'sans-serif'],
        proza: ['Proza Libre', 'sans-serif'],
        montserrat: ['Montserrat', 'sans-serif'],
      },
      backgroundImage: {
        'login': "url('/static/assets/bg-login.png')",
        'register': "url('/static/assets/bg-register.png')"
      }
    },
  },
  plugins: [
    require("flowbite/plugin")
  ],
}
